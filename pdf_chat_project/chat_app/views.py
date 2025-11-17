import os
import time
import logging
import openai
import asyncio
from concurrent.futures import ThreadPoolExecutor
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.views.decorators.http import require_POST, require_GET
from asgiref.sync import sync_to_async, async_to_sync

from .models import PDFDocument, ChatHistory
from .utils import (
    parse_pdf_ultra_fast,
    chunk_documents_ultra_fast,
    build_vector_store_fast,
    build_bm25_fast,
    hybrid_retrieval_optimized,
    answer_with_context_optimized,
    load_faiss_vector_store,
    save_bm25_object,
    load_bm25_object,
    save_tokenized_texts,
    load_tokenized_texts,
    save_docs,
    load_docs,
    get_s3_client,
    wait_for_s3_file
)

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=2)

# In-memory store for processed PDF data (for demonstration, will need persistence for production)
# Keyed by PDFDocument.id
processed_data_store = {}

def index(request):
    """Renders the main page with PDF upload and chat interface."""
    async def _index():
        pdfs = await sync_to_async(list)(PDFDocument.objects.all().order_by('-uploaded_at'))
        return render(request, 'chat_app/index.html', {'pdfs': pdfs})
    
    return async_to_sync(_index)()


async def process_pdf_in_background(pdf_doc_id):
    """Background processing function for PDF."""
    try:
        # Set tokenizers parallelism to avoid warnings
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Get PDF document
        pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_doc_id)
        
        # Update status to processing
        pdf_doc.processing_status = 'processing'
        pdf_doc.progress_percentage = 10
        await sync_to_async(pdf_doc.save)()
        
        start_time = time.time()
        
        # S3 file availability check for UAT_AWS
        if settings.ENVIRONMENT == 'UAT_AWS':
            s3_bucket = settings.AWS_STORAGE_BUCKET_NAME
            logger.info(f"Using S3 bucket: {s3_bucket}")

            # Check if file exists in S3
            storage_exists = await sync_to_async(pdf_doc.file.storage.exists)(pdf_doc.file.name)
            
            if not storage_exists:
                logger.error(f"S3 upload failed: File {pdf_doc.file.name} not found in bucket {s3_bucket}.")
                pdf_doc.processing_status = 'failed'
                pdf_doc.error_message = 'PDF file not found in S3 storage'
                await sync_to_async(pdf_doc.save)()
                return

            # Wait for the S3 file to become available
            s3_available = await wait_for_s3_file(pdf_doc.file.storage, pdf_doc.file.name)
            if not s3_available:
                logger.error(f"S3 object '{pdf_doc.file.name}' did not become available for download.")
                pdf_doc.processing_status = 'failed'
                pdf_doc.error_message = 'PDF file not accessible in S3 storage'
                await sync_to_async(pdf_doc.save)()
                return
        
        # Step 1: Parse PDF (30% progress)
        logger.info(f"Starting PDF parsing for: {pdf_doc.filename}")
        pages_markdown = await parse_pdf_ultra_fast(pdf_doc)
        pdf_doc.num_pages = len(pages_markdown)
        pdf_doc.progress_percentage = 30
        await sync_to_async(pdf_doc.save)()
        logger.info(f"PDF parsing completed for {pdf_doc.filename}. Pages: {pdf_doc.num_pages}")
        
        if not pages_markdown:
            pdf_doc.processing_status = 'failed'
            pdf_doc.error_message = 'PDF parsing resulted in no content'
            await sync_to_async(pdf_doc.save)()
            return
        
        # Step 2: Chunk documents (50% progress)
        logger.info(f"Chunking documents for: {pdf_doc.filename}")
        docs = await sync_to_async(chunk_documents_ultra_fast)(
            pages_markdown, pdf_doc.id, source=pdf_doc.filename
        )
        pdf_doc.num_chunks = len(docs)
        pdf_doc.progress_percentage = 50
        await sync_to_async(pdf_doc.save)()
        logger.info(f"Document chunking completed for {pdf_doc.filename}. Chunks: {pdf_doc.num_chunks}")
        
        # Step 3: Build vector store (80% progress)
        logger.info(f"Building vector store for: {pdf_doc.filename}")
        
        # Determine index base path
        if settings.ENVIRONMENT == 'UAT_AWS':
            index_base_path = f"faiss_indexes/{pdf_doc.id}"
        else:
            index_base_path = os.path.join(settings.MEDIA_ROOT, 'faiss_indexes', str(pdf_doc.id)).replace(" ", "_")
        
        # Build indexes in parallel using thread executor
        def build_vectorstore():
            return build_vector_store_fast(docs, pdf_doc.id, index_base_path)
        
        def build_bm25():
            return build_bm25_fast(docs)
        
        # Run both operations concurrently
        logger.info(f"Starting parallel vector store and BM25 building for PDF {pdf_doc.id}")
        vectorstore_task = asyncio.get_event_loop().run_in_executor(executor, build_vectorstore)
        bm25_task = asyncio.get_event_loop().run_in_executor(executor, build_bm25)
        
        vectorstore, (bm25_obj, tokenized_texts) = await asyncio.gather(vectorstore_task, bm25_task)
        logger.info(f"Vector store and BM25 building completed for PDF {pdf_doc.id}")
        
        pdf_doc.progress_percentage = 80
        await sync_to_async(pdf_doc.save)()
        
        # Step 4: Save everything (100% progress)
        logger.info(f"Saving processed data for: {pdf_doc.filename}")
        
        # Define paths for persistent storage
        bm25_file_path = os.path.join(index_base_path, 'bm25.pkl')
        tokenized_texts_file_path = os.path.join(index_base_path, 'tokenized_texts.pkl')
        docs_file_path = os.path.join(index_base_path, 'docs.pkl')
        
        # Save in thread executor to avoid blocking
        def save_all_data():
            save_bm25_object(bm25_obj, pdf_doc.id, bm25_file_path)
            save_tokenized_texts(tokenized_texts, pdf_doc.id, tokenized_texts_file_path)
            save_docs(docs, pdf_doc.id, docs_file_path)
            
            return bm25_file_path, tokenized_texts_file_path, docs_file_path
        
        bm25_path, tokenized_path, docs_path = await asyncio.get_event_loop().run_in_executor(
            executor, save_all_data
        )
        logger.info(f"Data saving completed for PDF {pdf_doc.id}")
        
        # Store processed data in memory for immediate use - IMPORTANT: Use correct key
        processed_data_store[pdf_doc.id] = {
            "docs": docs,
            "vectorstore": vectorstore,
            "bm25_obj": bm25_obj,
            "tokenized_texts": tokenized_texts,
        }
        logger.info(f"Stored data in memory for PDF {pdf_doc.id}. Memory store now has keys: {list(processed_data_store.keys())}")
        
        # Final update
        processing_time = time.time() - start_time
        pdf_doc.processing_time = processing_time
        pdf_doc.processed = True
        pdf_doc.processing_status = 'completed'
        pdf_doc.progress_percentage = 100
        pdf_doc.bm25_path = bm25_path
        pdf_doc.tokenized_texts_path = tokenized_path
        pdf_doc.docs_path = docs_path
        await sync_to_async(pdf_doc.save)()
        
        logger.info(f"PDF processing completed: {pdf_doc.filename} in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"PDF processing failed for ID {pdf_doc_id}: {e}", exc_info=True)
        try:
            pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_doc_id)
            pdf_doc.processing_status = 'failed'
            pdf_doc.error_message = str(e)
            await sync_to_async(pdf_doc.save)()
        except:
            pass

@require_POST
@csrf_exempt
def upload_pdf(request):
    """Handles PDF file upload and initiates background processing."""
    try:
        if 'pdf_file' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': 'No PDF file uploaded.'}, status=400)

        uploaded_file = request.FILES['pdf_file']
        
        # Create PDF document synchronously
        pdf_doc = PDFDocument.objects.create(
            filename=uploaded_file.name,
            processed=False,
            processing_status='uploading',
            progress_percentage=0
        )
        
        # Save file synchronously
        pdf_doc.file = uploaded_file
        pdf_doc.save()
        
        logger.info(f"PDF uploaded successfully: {pdf_doc.filename} (ID: {pdf_doc.id})")
        
        # Start background processing using thread executor to avoid event loop issues
        from threading import Thread
        def start_background_processing():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_pdf_in_background(pdf_doc.id))
            loop.close()
        
        thread = Thread(target=start_background_processing)
        thread.daemon = True
        thread.start()
        
        return JsonResponse({
            'status': 'success',
            'message': f'PDF "{uploaded_file.name}" uploaded successfully. Processing started.',
            'pdf_id': pdf_doc.id,
            'filename': pdf_doc.filename,
            'processing_status': 'processing',  # Add this for frontend
            'progress_percentage': 0,           # Add this for frontend
            'display_status': 'processing'      # Add this for frontend display
        })
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'Error uploading PDF: {str(e)}'}, status=500)

@require_POST
@csrf_exempt
def chat(request):
    """Handles chat queries for a processed PDF."""
    async def _chat():
        pdf_id = request.POST.get('pdf_id')
        query = request.POST.get('query')

        if not pdf_id or not query:
            return JsonResponse({'status': 'error', 'message': 'Missing pdf_id or query.'}, status=400)

        try:
            # Convert pdf_id to int if it's a string
            try:
                pdf_id = int(pdf_id)
            except (ValueError, TypeError):
                return JsonResponse({'status': 'error', 'message': 'Invalid pdf_id format.'}, status=400)

            pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_id)
            
            # More detailed status checking
            logger.info(f"Chat request for PDF {pdf_id}: processed={pdf_doc.processed}, processing_status={pdf_doc.processing_status}, progress={getattr(pdf_doc, 'progress_percentage', 0)}%, has_memory_data={pdf_id in processed_data_store}")
            
            # Check processing status more flexibly
            is_processing_complete = (
                pdf_doc.processed or 
                pdf_doc.processing_status == 'completed' or
                (pdf_doc.processing_status == 'processing' and pdf_doc.progress_percentage == 100)
            )
            
            # If not processed but has data in memory, allow chat
            if not is_processing_complete and pdf_id not in processed_data_store:
                logger.warning(f"PDF {pdf_id} not ready for chat. Status: {pdf_doc.processing_status}, Progress: {getattr(pdf_doc, 'progress_percentage', 0)}%")
                return JsonResponse({
                    'status': 'error', 
                    'message': f'PDF not yet processed. Current status: {pdf_doc.processing_status}',
                    'processing_status': pdf_doc.processing_status,
                    'progress': getattr(pdf_doc, 'progress_percentage', 0)
                }, status=400)

            # Retrieve processed data from store or load from disk
            data = processed_data_store.get(pdf_id)
            if not data:
                logger.info(f"Loading processed data from storage for PDF {pdf_id}")
                
                # Check if we have the required paths
                if not all([pdf_doc.bm25_path, pdf_doc.tokenized_texts_path, pdf_doc.docs_path]):
                    return JsonResponse({'status': 'error', 'message': 'Processed data paths not found. Please reprocess the PDF.'}, status=404)
                
                try:
                    # Load all components
                    vectorstore = await sync_to_async(load_faiss_vector_store)(
                        pdf_doc.id,
                        os.path.dirname(pdf_doc.bm25_path) if settings.ENVIRONMENT != 'UAT_AWS' else ""
                    )
                    bm25_obj = await sync_to_async(load_bm25_object)(pdf_doc.id, pdf_doc.bm25_path)
                    tokenized_texts = await sync_to_async(load_tokenized_texts)(pdf_doc.id, pdf_doc.tokenized_texts_path)
                    docs = await sync_to_async(load_docs)(pdf_doc.id, pdf_doc.docs_path)

                    if vectorstore is None or bm25_obj is None or not tokenized_texts or not docs:
                        raise ValueError("Failed to load all processed data components.")

                    # Store in memory for subsequent requests
                    processed_data_store[pdf_id] = {
                        "docs": docs,
                        "vectorstore": vectorstore,
                        "bm25_obj": bm25_obj,
                        "tokenized_texts": tokenized_texts,
                    }
                    logger.info(f"Successfully loaded processed data for PDF {pdf_id}")
                    
                except Exception as e:
                    logger.error(f"Error loading processed data from disk/S3 for PDF {pdf_id}: {e}", exc_info=True)
                    return JsonResponse({'status': 'error', 'message': f'Error loading processed data: {e}. Please reprocess the PDF.'}, status=500)

            # Get the processed data
            docs = processed_data_store[pdf_id]["docs"]
            vectorstore = processed_data_store[pdf_id]["vectorstore"]
            bm25_obj = processed_data_store[pdf_id]["bm25_obj"]
            tokenized_texts = processed_data_store[pdf_id]["tokenized_texts"]

            start_retrieval = time.time()
            candidates = hybrid_retrieval_optimized(
                query,
                docs,
                vectorstore,
                bm25_obj,
                tokenized_texts
            )
            retrieval_time = time.time() - start_retrieval

            start_answer = time.time()
            answer, chunk_data = await answer_with_context_optimized(query, candidates)
            answer_time = time.time() - start_answer

            # Save chat history
            await sync_to_async(ChatHistory.objects.create)(
                pdf_document=pdf_doc,
                question=query,
                answer=answer
            )
            
            # Format chunk_data for the frontend, including page details
            formatted_chunks = []
            for chunk in chunk_data:
                metadata = chunk.get('metadata', {})
                pages_info = metadata.get('pages', [metadata.get('page', 'N/A')])
                
                if isinstance(pages_info, list) and pages_info:
                    if len(pages_info) == 1:
                        pages_str = f"Page {pages_info[0]}"
                    else:
                        pages_str = f"Pages {min(pages_info)}-{max(pages_info)}"
                else:
                    pages_str = str(pages_info)

                formatted_chunks.append({
                    "page": pages_str,
                    "content": chunk.get('page_content', ''),
                    "source": metadata.get('source', 'Unknown'),
                    "chunk_length": metadata.get('chunk_length', 0),
                    "original_page": metadata.get('page', 'N/A') # Keep original page for specific chunk summary
                })

            return JsonResponse({
                'status': 'success',
                'answer': answer,
                'retrieval_time': f"{retrieval_time:.2f}s",
                'generation_time': f"{answer_time:.2f}s",
                'context': formatted_chunks # Use the formatted chunk_data
            })

        except PDFDocument.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'PDF document not found.'}, status=404)
        except Exception as e:
            logger.error(f"Error during chat: {e}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': f'Error during chat: {e}'}, status=500)
    
    return async_to_sync(_chat)()

@require_POST
@csrf_exempt
def summarize_chunk(request):
    """Generates a summary for a specific chunk of text."""
    async def _summarize_chunk():
        pdf_id = request.POST.get('pdf_id')
        chunk_content = request.POST.get('chunk_content')
        chunk_page = request.POST.get('chunk_page')
        query = request.POST.get('query') # New: Get the original query

        logger.info(f"summarize_chunk received: pdf_id={pdf_id}, chunk_page={chunk_page}, query={query}, chunk_content_len={len(chunk_content) if chunk_content else 0}")

        if not pdf_id or not chunk_content or not chunk_page or not query:
            logger.error("Missing pdf_id, chunk_content, chunk_page, or query for summarize_chunk.")
            return JsonResponse({'status': 'error', 'message': 'Missing pdf_id, chunk_content, chunk_page, or query.'}, status=400)

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI API key not configured for summarize_chunk.")
            return JsonResponse({'status': 'error', 'message': '❌ OpenAI API key not configured.'}, status=500)

        logger.info(f"OpenAI API key loaded for summarization: {'sk-proj-...' + openai_api_key[-5:] if openai_api_key else 'None'}")

        try:
            async with openai.AsyncOpenAI(api_key=openai_api_key) as client:
                prompt = f"""Based EXCLUSIVELY on the following context, provide a concise and accurate summary of the text chunk in relation to the original question.
            
            Original Question: {query}
            
            Context Information (from Page: {chunk_page}):
            {chunk_content}
            
            Instructions:
            - Summarize the provided text chunk specifically answering or relating to the Original Question.
            - Use ONLY information from the provided Context Information.
            - If the context doesn't contain relevant information for the question, state "This chunk does not contain information relevant to the question."
            - Be precise and keep the summary focused.
            - Keep the summary to a maximum of 100 words.
            
            Summary:"""
                logger.info(f"Summarization prompt for PDF ID {pdf_id}, Page {chunk_page}: {prompt[:500]}...") # Log first 500 chars of prompt

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a precise assistant that summarizes text chunks based on a given question and context. Never hallucinate or use external knowledge."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200,
                )
                summary = response.choices[0].message.content
                logger.info(f"Successfully generated summary for PDF ID {pdf_id}, Page {chunk_page}. Summary: {summary[:200]}...") # Log first 200 chars of summary
                return JsonResponse({'status': 'success', 'summary': summary})

        except openai.APIStatusError as e:
            logger.error(f"OpenAI API Status Error during chunk summarization (PDF ID {pdf_id}, Page {chunk_page}): Status {e.status_code}, Response: {e.response}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': f'OpenAI API error: {e.status_code} - {e.message}'}, status=500)
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API Connection Error during chunk summarization (PDF ID {pdf_id}, Page {chunk_page}): {e}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': f'OpenAI API connection error: {e}'}, status=500)
        except openai.RateLimitError as e:
            logger.error(f"OpenAI Rate Limit Error during chunk summarization (PDF ID {pdf_id}, Page {chunk_page}): {e}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': f'OpenAI rate limit exceeded: {e}'}, status=429)
        except Exception as e:
            logger.error(f"Unexpected error during chunk summarization (PDF ID {pdf_id}, Page {chunk_page}): {e}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': f'Error generating summary: {e}'}, status=500)

    return async_to_sync(_summarize_chunk)()


@require_GET
def check_processing_status(request, pdf_id):
    """Check PDF processing status for async operations."""
    async def _check_processing_status():
        try:
            pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_id)
            
            # Check if data exists in memory
            has_memory_data = pdf_id in processed_data_store
            memory_data_keys = list(processed_data_store.get(pdf_id, {}).keys()) if has_memory_data else []
            
            return JsonResponse({
                'status': pdf_doc.processing_status,
                'progress': getattr(pdf_doc, 'progress_percentage', 0),
                'processed': pdf_doc.processed,
                'error': getattr(pdf_doc, 'error_message', None),
                'num_pages': getattr(pdf_doc, 'num_pages', None),
                'num_chunks': getattr(pdf_doc, 'num_chunks', None),
                'filename': pdf_doc.filename,
                'has_memory_data': has_memory_data,
                'memory_data_keys': memory_data_keys,
                'processing_time': getattr(pdf_doc, 'processing_time', None),
                'bm25_path': getattr(pdf_doc, 'bm25_path', None),
                'tokenized_texts_path': getattr(pdf_doc, 'tokenized_texts_path', None),
                'docs_path': getattr(pdf_doc, 'docs_path', None),
            })
        except PDFDocument.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'PDF not found'}, status=404)
        except Exception as e:
            logger.error(f"Error checking processing status: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return async_to_sync(_check_processing_status)()

@require_GET
def get_chat_history(request, pdf_id):
    """Retrieves chat history for a given PDF."""
    async def _get_chat_history():
        try:
            pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_id)
            
            # Get chat history using sync_to_async properly
            chat_history_qs = ChatHistory.objects.filter(pdf_document=pdf_doc).order_by('timestamp')
            chat_history = await sync_to_async(list)(chat_history_qs)
            
            chat_entries = []
            for entry in chat_history:
                chat_entries.append({
                    'question': entry.question,
                    'answer': entry.answer,
                    'timestamp': entry.timestamp.isoformat()
                })
            
            return JsonResponse({'status': 'success', 'history': chat_entries})
        except PDFDocument.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'PDF document not found.'}, status=404)
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': f'Error retrieving chat history: {e}'}, status=500)
    
    return async_to_sync(_get_chat_history)()

@require_POST
@csrf_exempt
def clear_data(request):
    """Clears all processed data and PDF documents."""
    async def _clear_data():
        try:
            # Clear in-memory store
            processed_data_store.clear()

            # Delete all PDF documents and their associated files/indexes
            pdf_docs_to_delete = await sync_to_async(list)(PDFDocument.objects.all())
            s3_client = None
            if settings.ENVIRONMENT == 'UAT_AWS':
                s3_client = get_s3_client()
                s3_bucket = settings.AWS_STORAGE_BUCKET_NAME

            for pdf_doc in pdf_docs_to_delete:
                # Delete PDF file from storage (local or S3)
                if pdf_doc.file:
                    await sync_to_async(pdf_doc.file.delete)(save=False)

                # Delete FAISS index and related files
                if settings.ENVIRONMENT == 'UAT_AWS' and s3_client:
                    # List and delete all objects under the PDF's FAISS index prefix
                    prefix = f"faiss_indexes/{pdf_doc.id}/"
                    try:
                        response = await sync_to_async(s3_client.list_objects_v2)(Bucket=s3_bucket, Prefix=prefix)
                        if 'Contents' in response:
                            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                            if objects_to_delete:
                                await sync_to_async(s3_client.delete_objects)(
                                    Bucket=s3_bucket,
                                    Delete={'Objects': objects_to_delete, 'Quiet': True}
                                )
                                logger.info(f"Deleted S3 objects for PDF ID {pdf_doc.id} under prefix {prefix}")
                    except Exception as e:
                        logger.error(f"Error deleting S3 objects for PDF ID {pdf_doc.id}: {e}")
                else:
                    # Local deletion
                    index_dir_path = os.path.join(settings.MEDIA_ROOT, 'faiss_indexes', str(pdf_doc.id))
                    if os.path.isdir(index_dir_path):
                        import shutil
                        await sync_to_async(shutil.rmtree)(index_dir_path)
                        logger.info(f"Deleted local FAISS index directory: {index_dir_path}")

            # Delete all database entries
            def delete_all_data():
                PDFDocument.objects.all().delete()
                ChatHistory.objects.all().delete()
            
            await sync_to_async(delete_all_data)()

            # Clear local media root directories if empty (only relevant for local storage)
            if settings.ENVIRONMENT != 'UAT_AWS':
                def cleanup_local_dirs():
                    media_root = settings.MEDIA_ROOT
                    if os.path.exists(media_root) and not os.listdir(media_root):
                        os.rmdir(media_root)

                    faiss_indexes_root = os.path.join(media_root, 'faiss_indexes')
                    if os.path.exists(faiss_indexes_root) and not os.listdir(faiss_indexes_root):
                        os.rmdir(faiss_indexes_root)
                
                await sync_to_async(cleanup_local_dirs)()

            return JsonResponse({'status': 'success', 'message': 'All data and cache cleared.'})
        except Exception as e:
            logger.error(f"Error clearing data: {e}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': f'Error clearing data: {e}'}, status=500)
    
    return async_to_sync(_clear_data)()

@require_GET
def debug_pdf_status(request, pdf_id):
    """Debug endpoint to check PDF status and data availability."""
    async def _debug_pdf_status():
        try:
            pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_id)
            
            # Check memory store
            has_memory_data = int(pdf_id) in processed_data_store
            memory_data_keys = list(processed_data_store.get(int(pdf_id), {}).keys()) if has_memory_data else []
            
            return JsonResponse({
                'pdf_id': pdf_id,
                'filename': pdf_doc.filename,
                'processed': pdf_doc.processed,
                'processing_status': pdf_doc.processing_status,
                'progress_percentage': getattr(pdf_doc, 'progress_percentage', 0),
                'error_message': getattr(pdf_doc, 'error_message', None),
                'bm25_path': getattr(pdf_doc, 'bm25_path', None),
                'tokenized_texts_path': getattr(pdf_doc, 'tokenized_texts_path', None),
                'docs_path': getattr(pdf_doc, 'docs_path', None),
                'has_memory_data': has_memory_data,
                'memory_data_keys': memory_data_keys,
                'num_pages': getattr(pdf_doc, 'num_pages', 0),
                'num_chunks': getattr(pdf_doc, 'num_chunks', 0),
            })
        except PDFDocument.DoesNotExist:
            return JsonResponse({'error': 'PDF not found'}, status=404)
        except Exception as e:
            logger.error(f"Debug error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return async_to_sync(_debug_pdf_status)()

@require_GET
def list_pdfs(request):
    """List all PDFs with their status for debugging."""
    try:
        pdfs = PDFDocument.objects.all().order_by('-id')
        pdf_list = []
        
        for pdf in pdfs:
            pdf_list.append({
                'id': pdf.id,
                'filename': pdf.filename,
                'processed': pdf.processed,
                'processing_status': getattr(pdf, 'processing_status', 'unknown'),
                'progress_percentage': getattr(pdf, 'progress_percentage', 0),
                'has_memory_data': pdf.id in processed_data_store,
                'num_pages': getattr(pdf, 'num_pages', 0),
                'num_chunks': getattr(pdf, 'num_chunks', 0),
                'error_message': getattr(pdf, 'error_message', None),
                'uploaded_at': pdf.uploaded_at.isoformat() if hasattr(pdf, 'uploaded_at') else None
            })
        
        return JsonResponse({
            'pdfs': pdf_list,
            'memory_store_keys': list(processed_data_store.keys()),
            'total_pdfs': len(pdf_list)
        })
        
    except Exception as e:
        logger.error(f"Error listing PDFs: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@require_GET
def quick_status(request, pdf_id):
    """Quick synchronous status check for a PDF."""
    try:
        pdf_doc = PDFDocument.objects.get(id=pdf_id)
        
        return JsonResponse({
            'pdf_id': pdf_id,
            'filename': pdf_doc.filename,
            'processed': pdf_doc.processed,
            'processing_status': getattr(pdf_doc, 'processing_status', 'unknown'),
            'progress_percentage': getattr(pdf_doc, 'progress_percentage', 0),
            'num_pages': getattr(pdf_doc, 'num_pages', None),
            'num_chunks': getattr(pdf_doc, 'num_chunks', None),
            'error_message': getattr(pdf_doc, 'error_message', None),
            'has_memory_data': pdf_id in processed_data_store,
            'memory_store_keys': list(processed_data_store.keys()),
            'is_truly_ready': (
                pdf_doc.processed and 
                pdf_doc.processing_status == 'completed' and 
                getattr(pdf_doc, 'num_pages', 0) > 0 and 
                getattr(pdf_doc, 'num_chunks', 0) > 0
            )
        })
        
    except PDFDocument.DoesNotExist:
        return JsonResponse({'error': 'PDF not found'}, status=404)
    except Exception as e:
        logger.error(f"Quick status error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@require_GET
def pdf_status_simple(request, pdf_id):
    """Simple endpoint to get PDF status for frontend display."""
    try:
        pdf_doc = PDFDocument.objects.get(id=pdf_id)
        
        # Determine the actual status for frontend display
        is_ready_for_chat = (
            pdf_doc.processed and 
            pdf_doc.processing_status == 'completed' and 
            getattr(pdf_doc, 'num_pages', 0) > 0 and 
            getattr(pdf_doc, 'num_chunks', 0) > 0
        )
        
        # Frontend display status
        if pdf_doc.processing_status == 'failed':
            display_status = 'failed'
        elif is_ready_for_chat:
            display_status = 'processed'
        elif pdf_doc.processing_status in ['uploading', 'processing']:
            display_status = 'processing'
        else:
            display_status = 'unknown'
        
        return JsonResponse({
            'pdf_id': pdf_id,
            'filename': pdf_doc.filename,
            'display_status': display_status,  # Use this for frontend display
            'is_ready_for_chat': is_ready_for_chat,
            'processing_status': getattr(pdf_doc, 'processing_status', 'unknown'),
            'progress_percentage': getattr(pdf_doc, 'progress_percentage', 0),
            'num_pages': getattr(pdf_doc, 'num_pages', None),
            'num_chunks': getattr(pdf_doc, 'num_chunks', None),
            'error_message': getattr(pdf_doc, 'error_message', None),
        })
        
    except PDFDocument.DoesNotExist:
        return JsonResponse({'error': 'PDF not found'}, status=404)
    except Exception as e:
        logger.error(f"PDF status error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@require_GET
def get_all_pdfs_status(request):
    """Get all PDFs with their correct status for frontend display."""
    try:
        pdfs = PDFDocument.objects.all().order_by('-id')
        pdf_list = []
        
        for pdf in pdfs:
            # Determine the actual status for frontend display
            is_ready_for_chat = (
                pdf.processed and 
                getattr(pdf, 'processing_status', '') == 'completed' and 
                getattr(pdf, 'num_pages', 0) > 0 and 
                getattr(pdf, 'num_chunks', 0) > 0
            )
            
            # Frontend display status
            processing_status = getattr(pdf, 'processing_status', 'unknown')
            if processing_status == 'failed':
                display_status = 'failed'
            elif is_ready_for_chat:
                display_status = 'processed'
            elif processing_status in ['uploading', 'processing']:
                display_status = 'processing'
            else:
                display_status = 'unknown'
            
            pdf_list.append({
                'id': pdf.id,
                'filename': pdf.filename,
                'display_status': display_status,  # Frontend should use this
                'is_ready_for_chat': is_ready_for_chat,
                'processing_status': processing_status,
                'progress_percentage': getattr(pdf, 'progress_percentage', 0),
                'num_pages': getattr(pdf, 'num_pages', None),
                'num_chunks': getattr(pdf, 'num_chunks', None),
                'error_message': getattr(pdf, 'error_message', None),
                'uploaded_at': pdf.uploaded_at.isoformat() if hasattr(pdf, 'uploaded_at') else None
            })
        
        return JsonResponse({'pdfs': pdf_list})
        
    except Exception as e:
        logger.error(f"Error getting all PDFs status: {e}")
        return JsonResponse({'error': str(e)}, status=500)
