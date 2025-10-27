import os
import tempfile
import json
import asyncio
import time
import logging
import openai
from concurrent.futures import ThreadPoolExecutor

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage
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
    load_embedding_model, # Needed for loading FAISS index
    load_faiss_vector_store, # New function to load FAISS with deserialization allowed
    save_bm25_object,
    load_bm25_object,
    save_tokenized_texts,
    load_tokenized_texts,
    save_docs,
    load_docs
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# In-memory store for processed PDF data (for demonstration, will need persistence for production)
# Keyed by PDFDocument.id
processed_data_store = {}

def index(request):
    """Renders the main page with PDF upload and chat interface."""
    pdfs = PDFDocument.objects.all().order_by('-uploaded_at')
    return render(request, 'chat_app/index.html', {'pdfs': pdfs})

@require_POST
@csrf_exempt
async def upload_pdf(request):
    """Handles PDF file upload and initiates processing."""
    if 'pdf_file' not in request.FILES:
        return JsonResponse({'status': 'error', 'message': 'No PDF file uploaded.'}, status=400)

    uploaded_file = request.FILES['pdf_file']
    fs = FileSystemStorage(location=settings.MEDIA_ROOT)

    # Save the uploaded file temporarily
    filename = await sync_to_async(fs.save)(uploaded_file.name, uploaded_file)
    pdf_path = os.path.join(settings.MEDIA_ROOT, filename)

    try:
        # Create a new PDFDocument entry
        pdf_doc = await sync_to_async(PDFDocument.objects.create)(
            file=filename,
            filename=uploaded_file.name,
            processed=False
        )

        # Asynchronous processing
        start_total = time.time()

        # 1. Fast PDF parsing
        pages_markdown = await parse_pdf_ultra_fast(pdf_path)
        pdf_doc.num_pages = len(pages_markdown)

        # 2. Fast chunking with deduplication
        docs = chunk_documents_ultra_fast(pages_markdown, source=uploaded_file.name)
        pdf_doc.num_chunks = len(docs)

        # 3. Parallel indexing
        # Use a unique index directory for each PDF, replacing spaces for FAISS compatibility
        index_dir_path = os.path.join(settings.MEDIA_ROOT, 'faiss_indexes', str(pdf_doc.id)).replace(" ", "_")

        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(build_bm25_fast, docs)
            vector_future = executor.submit(build_vector_store_fast, docs, index_dir_path)

            bm25_obj, tokenized_texts = await sync_to_async(bm25_future.result)()
            vectorstore = await sync_to_async(vector_future.result)()

        total_time = time.time() - start_total
        pdf_doc.processing_time = total_time
        pdf_doc.processed = True
        await sync_to_async(pdf_doc.save)()

        # Define paths for persistent storage
        bm25_file_path = os.path.join(index_dir_path, 'bm25.pkl')
        tokenized_texts_file_path = os.path.join(index_dir_path, 'tokenized_texts.json')
        docs_file_path = os.path.join(index_dir_path, 'docs.json')

        # Save BM25, tokenized_texts, and docs to disk
        await sync_to_async(save_bm25_object)(bm25_obj, bm25_file_path)
        await sync_to_async(save_tokenized_texts)(tokenized_texts, tokenized_texts_file_path)
        await sync_to_async(save_docs)(docs, docs_file_path)

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
        # Clean up the uploaded file if processing fails
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
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
                    vectorstore = await sync_to_async(load_faiss_vector_store)(os.path.dirname(pdf_doc.bm25_path))
                    bm25_obj = await sync_to_async(load_bm25_object)(pdf_doc.bm25_path)
                    tokenized_texts = await sync_to_async(load_tokenized_texts)(pdf_doc.tokenized_texts_path)
                    docs = await sync_to_async(load_docs)(pdf_doc.docs_path)

                    # Store in memory for subsequent requests
                    processed_data_store[pdf_doc.id] = {
                        "docs": docs,
                        "vectorstore": vectorstore,
                        "bm25_obj": bm25_obj,
                        "tokenized_texts": tokenized_texts,
                    }
                except Exception as e:
                    logger.error(f"Error loading processed data from disk for PDF {pdf_id}: {e}", exc_info=True)
                    return JsonResponse({'status': 'error', 'message': f'Error loading processed data: {e}'}, status=500)
            else:
                return JsonResponse({'status': 'error', 'message': 'Processed data not found (neither in memory nor on disk).'}, status=404)

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

    if not pdf_id or not chunk_content or not chunk_page or not query:
        return JsonResponse({'status': 'error', 'message': 'Missing pdf_id, chunk_content, chunk_page, or query.'}, status=400)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not configured for summarize_chunk.")
        return JsonResponse({'status': 'error', 'message': '❌ OpenAI API key not configured.'}, status=500)

    logger.info(f"OpenAI API key loaded for summarization: {'sk-proj-...' + openai_api_key[-5:] if openai_api_key else 'None'}")

    try:
        client = openai.AsyncOpenAI(api_key=openai_api_key)
        
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
        return JsonResponse({'status': 'success', 'summary': summary})

    except Exception as e:
        logger.error(f"OpenAI API error during chunk summarization: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'Error generating summary: {e}'}, status=500)


@require_GET
async def get_chat_history(request, pdf_id):
    """Retrieves chat history for a given PDF."""
    try:
        pdf_doc = await sync_to_async(PDFDocument.objects.get)(id=pdf_id)
        history = await sync_to_async(ChatHistory.objects.filter)(pdf_document=pdf_doc)
        history = await sync_to_async(history.order_by)('timestamp')
        chat_entries = []
        async for entry in history:
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
        for pdf_doc in pdf_docs_to_delete:
            # Delete PDF file
            if pdf_doc.file:
                await sync_to_async(pdf_doc.file.delete)(save=False) # delete file from storage

            # Delete FAISS index directory
            index_dir_path = os.path.join(settings.MEDIA_ROOT, 'faiss_indexes', str(pdf_doc.id))
            if os.path.isdir(index_dir_path):
                import shutil
                await sync_to_async(shutil.rmtree)(index_dir_path)

        await sync_to_async(PDFDocument.objects.all().delete)()
        await sync_to_async(ChatHistory.objects.all().delete)()

        # Clear media root directories if empty
        media_root = settings.MEDIA_ROOT
        if os.path.exists(media_root) and not os.listdir(media_root):
            await sync_to_async(os.rmdir)(media_root)

        faiss_indexes_root = os.path.join(media_root, 'faiss_indexes')
        if os.path.exists(faiss_indexes_root) and not os.listdir(faiss_indexes_root):
            await sync_to_async(os.rmdir)(faiss_indexes_root)

        return JsonResponse({'status': 'success', 'message': 'All data and cache cleared.'})
    except Exception as e:
        logger.error(f"Error clearing data: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'Error clearing data: {e}'}, status=500)
