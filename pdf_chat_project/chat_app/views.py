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
from asgiref.sync import sync_to_async

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

async def index(request):
    """Renders the main page with PDF upload and chat interface."""
    pdfs = await sync_to_async(list)(PDFDocument.objects.all().order_by('-uploaded_at'))
    return render(request, 'chat_app/index.html', {'pdfs': pdfs})


async def process_pdf_in_background(pdf_doc_id):
    """Background processing function for PDF."""
    try:
        # Get PDF document
        pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_doc_id)
        
        # Update status to processing
        pdf_doc.processing_status = 'processing'
        pdf_doc.progress_percentage = 10
        await sync_to_async(pdf_doc.save)()
        
        start_time = time.time()
        
        # Step 1: Parse PDF (30% progress)
        logger.info(f"Starting PDF parsing for: {pdf_doc.filename}")
        pages_markdown = await parse_pdf_ultra_fast(pdf_doc)
        pdf_doc.num_pages = len(pages_markdown)
        pdf_doc.progress_percentage = 30
        await sync_to_async(pdf_doc.save)()
        
        # Step 2: Chunk documents (50% progress)
        logger.info(f"Chunking documents for: {pdf_doc.filename}")
        docs = await sync_to_async(chunk_documents_ultra_fast)(
            pages_markdown, pdf_doc.id, source=pdf_doc.filename
        )
        pdf_doc.num_chunks = len(docs)
        pdf_doc.progress_percentage = 50
        await sync_to_async(pdf_doc.save)()
        
        # Step 3: Build vector store (80% progress)
        logger.info(f"Building vector store for: {pdf_doc.filename}")
        
        # Determine index base path
        if settings.ENVIRONMENT == 'UAT_AWS':
            index_base_path = f"faiss_indexes/{pdf_doc.id}"
        else:
            index_base_path = os.path.join(settings.MEDIA_ROOT, 'faiss_indexes', str(pdf_doc.id)).replace(" ", "_")
        
        # Build indexes in parallel using thread executor
        def build_vectorstore():
            return build_vector_store_fast(docs, pdf_doc.id)
        
        def build_bm25():
            return build_bm25_fast(docs)
        
        # Run both operations concurrently
        vectorstore_task = asyncio.get_event_loop().run_in_executor(executor, build_vectorstore)
        bm25_task = asyncio.get_event_loop().run_in_executor(executor, build_bm25)
        
        vectorstore, (bm25_obj, tokenized_texts) = await asyncio.gather(vectorstore_task, bm25_task)
        
        pdf_doc.progress_percentage = 80
        await sync_to_async(pdf_doc.save)()
        
        # Step 4: Save everything (100% progress)
        logger.info(f"Saving processed data for: {pdf_doc.filename}")
        
        # Save in thread executor to avoid blocking
        def save_all_data():
            bm25_file_path = os.path.join(index_base_path, 'bm25.pkl')
            tokenized_texts_file_path = os.path.join(index_base_path, 'tokenized_texts.pkl')
            docs_file_path = os.path.join(index_base_path, 'docs.pkl')
            
            save_bm25_object(bm25_obj, bm25_file_path)
            save_tokenized_texts(tokenized_texts, tokenized_texts_file_path)
            save_docs(docs, docs_file_path)
            
            return bm25_file_path, tokenized_texts_file_path, docs_file_path
        
        bm25_path, tokenized_path, docs_path = await asyncio.get_event_loop().run_in_executor(
            executor, save_all_data
        )
        
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
        logger.error(f"PDF processing failed for ID {pdf_doc_id}: {e}")
        try:
            pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_doc_id)
            pdf_doc.processing_status = 'failed'
            pdf_doc.error_message = str(e)
            await sync_to_async(pdf_doc.save)()
        except:
            pass

@require_POST
@csrf_exempt
async def upload_pdf(request):
    """Handles PDF file upload and initiates processing."""
    if 'pdf_file' not in request.FILES:
        return JsonResponse({'status': 'error', 'message': 'No PDF file uploaded.'}, status=400)

    uploaded_file = request.FILES['pdf_file']

    # Create a new PDFDocument instance without saving the file yet
    pdf_doc = await sync_to_async(PDFDocument.objects.create)(
        filename=uploaded_file.name,
        processed=False
    )
    logger.info(f"PDFDocument instance created (ID: {pdf_doc.id}).")

    # Assign the uploaded file to the FileField and save to trigger storage backend
    pdf_doc.file = uploaded_file
    await sync_to_async(pdf_doc.save)()
    logger.info(f"PDFDocument saved. File name in DB: {pdf_doc.file.name}, URL: {pdf_doc.file.url}") # Log S3 object key and URL
    
    if settings.ENVIRONMENT == 'UAT_AWS':
        s3_bucket = settings.AWS_STORAGE_BUCKET_NAME
        print(f"DEBUG (views.py): ENVIRONMENT is UAT_AWS. Using S3 bucket: {s3_bucket}")
        logger.info(f"Using S3 bucket: {s3_bucket}")

        # Use django-storages's own storage backend to check for file existence
        # This is more reliable as it uses the same logic that handled the upload.
        # The pdf_doc.file.name already contains the path relative to the bucket root (e.g., 'pdfs/filename.pdf')
        print(f"DEBUG (views.py): Checking S3 existence for {pdf_doc.file.name} via storage backend.")
        storage_exists = await sync_to_async(pdf_doc.file.storage.exists)(pdf_doc.file.name)
        
        if not storage_exists:
            print(f"ERROR (views.py): S3 upload failed: File {pdf_doc.file.name} not found by storage backend in bucket {s3_bucket}.")
            logger.error(f"S3 upload failed: File {pdf_doc.file.name} not found by storage backend in bucket {s3_bucket}.")
            await sync_to_async(pdf_doc.delete)() # Clean up DB entry
            return JsonResponse({'status': 'error', 'message': 'Failed to upload PDF to S3.'}, status=500)
        print(f"DEBUG (views.py): S3 upload confirmed for {pdf_doc.file.name} using storage.exists().")
        logger.info(f"S3 upload confirmed for {pdf_doc.file.name} using storage.exists().")

        # Wait for the S3 file to become available before proceeding with parsing
        # Use the storage backend's exists method for consistency with upload
        print(f"DEBUG (views.py): Waiting for S3 object '{pdf_doc.file.name}' to be available via storage backend.")
        s3_available = await wait_for_s3_file(pdf_doc.file.storage, pdf_doc.file.name)
        if not s3_available:
            print(f"ERROR (views.py): S3 object '{pdf_doc.file.name}' did not become available for download.")
            await sync_to_async(pdf_doc.delete)() # Clean up DB entry
            return JsonResponse({'status': 'error', 'message': 'Failed to confirm PDF availability in S3.'}, status=500)
        print(f"DEBUG (views.py): S3 object '{pdf_doc.file.name}' is now available.")

    try:
        start_total = time.time()
        print(f"DEBUG (views.py): Starting PDF processing for PDF ID: {pdf_doc.id}")

        # 1. Fast PDF parsing (utils.parse_pdf_ultra_fast now handles S3 download internally)
        print(f"DEBUG (views.py): Calling parse_pdf_ultra_fast for PDF ID: {pdf_doc.id}")
        pages_markdown = await parse_pdf_ultra_fast(pdf_doc)
        pdf_doc.num_pages = len(pages_markdown)
        print(f"DEBUG (views.py): PDF parsed. Number of pages: {pdf_doc.num_pages}")

        # 2. Fast chunking with deduplication
        print(f"DEBUG (views.py): Chunking documents for PDF ID: {pdf_doc.id}")
        docs = chunk_documents_ultra_fast(pages_markdown, pdf_doc.id, source=uploaded_file.name)
        pdf_doc.num_chunks = len(docs)
        print(f"DEBUG (views.py): Documents chunked. Number of chunks: {pdf_doc.num_chunks}")

        # 3. Parallel indexing
        # Determine the base path for index files (local or S3 object key prefix)
        if settings.ENVIRONMENT == 'UAT_AWS':
            # For S3, paths are object keys, not file system paths
            index_base_path = f"faiss_indexes/{pdf_doc.id}"
            print(f"DEBUG (views.py): Index base path (S3): {index_base_path}")
        else:
            # For local, use MEDIA_ROOT
            index_base_path = os.path.join(settings.MEDIA_ROOT, 'faiss_indexes', str(pdf_doc.id)).replace(" ", "_")
            print(f"DEBUG (views.py): Index base path (Local): {index_base_path}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(build_bm25_fast, docs)
            vector_future = executor.submit(build_vector_store_fast, docs, pdf_doc.id, index_base_path)

            bm25_obj, tokenized_texts = await asyncio.get_event_loop().run_in_executor(
                None, bm25_future.result
            )
            vectorstore = await asyncio.get_event_loop().run_in_executor(
                None, vector_future.result
            )

        total_time = time.time() - start_total
        pdf_doc.processing_time = total_time
        pdf_doc.processed = True
        await sync_to_async(pdf_doc.save)()
        logger.info(f"PDFDocument saved. File URL: {pdf_doc.file.url}") # Log file URL

        if not pages_markdown:
            raise ValueError("PDF parsing resulted in no content.")

        # Define paths for persistent storage (these will be S3 object keys or local paths)
        bm25_file_path = os.path.join(index_base_path, 'bm25.pkl')
        tokenized_texts_file_path = os.path.join(index_base_path, 'tokenized_texts.json')
        docs_file_path = os.path.join(index_base_path, 'docs.json')

        # Save BM25, tokenized_texts, and docs using the updated utility functions
        await sync_to_async(save_bm25_object)(bm25_obj, pdf_doc.id, bm25_file_path)
        await sync_to_async(save_tokenized_texts)(tokenized_texts, pdf_doc.id, tokenized_texts_file_path)
        await sync_to_async(save_docs)(docs, pdf_doc.id, docs_file_path)

        # Update PDFDocument with paths
        pdf_doc.bm25_path = bm25_file_path
        pdf_doc.tokenized_texts_path = tokenized_texts_file_path
        pdf_doc.docs_path = docs_file_path
        await sync_to_async(pdf_doc.save)()

        # Store processed data in memory for immediate use (optional, can be loaded on demand)
        processed_data_store[pdf_doc.id] = {
            "docs": docs,
            "vectorstore": vectorstore,
            "bm25_obj": bm25_obj,
            "tokenized_texts": tokenized_texts,
        }

        return JsonResponse({
            'status': 'success',
            'message': f'PDF "{uploaded_file.name}" processed successfully in {total_time:.2f}s.',
            'pdf_id': pdf_doc.id,
            'num_pages': pdf_doc.num_pages,
            'num_chunks': pdf_doc.num_chunks,
            'processing_time': f"{total_time:.2f}s"
        })

    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        # Clean up the uploaded file if processing fails (only for local storage)
        try:
            if settings.ENVIRONMENT != 'UAT_AWS':
                # If a local file was saved to the model's FileField, remove it
                if 'pdf_doc' in locals() and getattr(pdf_doc, 'file', None):
                    try:
                        local_path = getattr(pdf_doc.file, 'path', None)
                        if local_path and os.path.exists(local_path):
                            os.remove(local_path)
                    except Exception:
                        pass
        except Exception:
            pass
        # If an error occurs after pdf_doc is created, ensure it's deleted
        if 'pdf_doc' in locals() and pdf_doc.pk:
            await sync_to_async(pdf_doc.delete)()
        return JsonResponse({'status': 'error', 'message': f'Error processing PDF: {e}'}, status=500)

@require_POST
@csrf_exempt
async def chat(request):
    """Handles chat queries for a processed PDF."""
    pdf_id = request.POST.get('pdf_id')
    query = request.POST.get('query')

    if not pdf_id or not query:
        return JsonResponse({'status': 'error', 'message': 'Missing pdf_id or query.'}, status=400)

    try:
        pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_id)
        if not pdf_doc.processed:
            return JsonResponse({'status': 'error', 'message': 'PDF not yet processed.'}, status=400)

        # Retrieve processed data from store or load from disk
        data = processed_data_store.get(pdf_id)
        if not data:
            if pdf_doc.bm25_path and pdf_doc.tokenized_texts_path and pdf_doc.docs_path:
                try:
                    # Pass pdf_id to loading functions
                    # For FAISS, the path is the directory containing index.faiss and index.pkl
                    # For S3, load_faiss_vector_store uses pdf_id to construct S3 paths
                    # For FAISS, the path is the directory containing index.faiss and index.pkl for local storage.
                    # For S3, load_faiss_vector_store uses pdf_id to construct S3 paths internally, so index_path can be an empty string.
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
                    processed_data_store[pdf_doc.id] = {
                        "docs": docs,
                        "vectorstore": vectorstore,
                        "bm25_obj": bm25_obj,
                        "tokenized_texts": tokenized_texts,
                    }
                except Exception as e:
                    logger.error(f"Error loading processed data from disk/S3 for PDF {pdf_id}: {e}", exc_info=True)
                    return JsonResponse({'status': 'error', 'message': f'Error loading processed data: {e}'}, status=500)
            else:
                return JsonResponse({'status': 'error', 'message': 'Processed data paths not found in PDFDocument.'}, status=404)

        docs = processed_data_store[pdf_doc.id]["docs"]
        vectorstore = processed_data_store[pdf_doc.id]["vectorstore"]
        bm25_obj = processed_data_store[pdf_doc.id]["bm25_obj"]
        tokenized_texts = processed_data_store[pdf_doc.id]["tokenized_texts"]

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

@require_POST
@csrf_exempt
async def summarize_chunk(request):
    """Generates a summary for a specific chunk of text."""
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


@require_GET
async def check_processing_status(request, pdf_id):
    """Check PDF processing status for async operations."""
    try:
        pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_id)
        return JsonResponse({
            'status': pdf_doc.processing_status,
            'progress': pdf_doc.progress_percentage,
            'processed': pdf_doc.processed,
            'error': pdf_doc.error_message,
            'num_pages': getattr(pdf_doc, 'num_pages', 0),
            'num_chunks': getattr(pdf_doc, 'num_chunks', 0),
            'filename': pdf_doc.filename
        })
    except PDFDocument.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'PDF not found'}, status=404)
    except Exception as e:
        logger.error(f"Error checking processing status: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@require_GET
async def get_chat_history(request, pdf_id):
    """Retrieves chat history for a given PDF."""
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

@require_POST
@csrf_exempt
async def clear_data(request):
    """Clears all processed data and PDF documents."""
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
