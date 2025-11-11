import os
import tempfile
import json
import asyncio
import time
import logging
import openai
import httpx  # Import httpx
from concurrent.futures import ThreadPoolExecutor

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.http import require_POST, require_GET
from asgiref.sync import sync_to_async
from botocore.exceptions import ClientError

from .models import PDFDocument, ChatHistory
from .utils import (
    parse_pdf_ultra_fast,
    chunk_documents_ultra_fast,
    build_vector_store_fast,
    build_bm25_fast,
    hybrid_retrieval_optimized,
    answer_with_context_optimized,
    load_embedding_model,  # Needed for loading FAISS index
    load_faiss_vector_store,  # New function to load FAISS with deserialization allowed
    save_bm25_object,
    load_bm25_object,
    save_tokenized_texts,
    load_tokenized_texts,
    save_docs,
    load_docs,
    get_faiss_index_s3_path,  # For S3 path generation
    get_s3_client,  # For S3 deletion
    s3_file_exists,  # For checking S3 file existence
    wait_for_s3_file,  # For waiting on S3 file availability
    get_pdf_s3_path  # For generating S3 path for PDF files
)
from langchain_core.documents import Document
from urllib.parse import urlparse

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
def upload_pdf(request):
    """Handles PDF file upload and initiates processing."""
    if 'pdf_file' not in request.FILES:
        return JsonResponse({'status': 'error', 'message': 'No PDF file uploaded.'}, status=400)

    uploaded_file = request.FILES['pdf_file']

    # Create a new PDFDocument instance without saving the file yet
    pdf_doc = PDFDocument.objects.create(
        filename=uploaded_file.name,
        processed=False
    )
    logger.info(f"PDFDocument instance created (ID: {pdf_doc.id}).")

    # Assign the uploaded file to the FileField and save to trigger storage backend
    pdf_doc.file = uploaded_file
    pdf_doc.save()
    # Log S3 object key and URL
    logger.info(
        f"PDFDocument saved. File name in DB: {pdf_doc.file.name}, URL: {pdf_doc.file.url}")

    if settings.ENVIRONMENT == 'UAT_AWS':
        s3_bucket = settings.AWS_STORAGE_BUCKET_NAME
        print(
            f"DEBUG (views.py): ENVIRONMENT is UAT_AWS. Using S3 bucket: {s3_bucket}")
        logger.info(f"Using S3 bucket: {s3_bucket}")

        print(
            f"DEBUG (views.py): Checking S3 existence for {pdf_doc.file.name} via storage backend.")
        storage_exists = pdf_doc.file.storage.exists(pdf_doc.file.name)

        if not storage_exists:
            print(
                f"ERROR (views.py): S3 upload failed: File {pdf_doc.file.name} not found by storage backend in bucket {s3_bucket}.")
            logger.error(
                f"S3 upload failed: File {pdf_doc.file.name} not found by storage backend in bucket {s3_bucket}.")
            pdf_doc.delete()  # Clean up DB entry
            return JsonResponse({'status': 'error', 'message': 'Failed to upload PDF to S3.'}, status=500)
        print(
            f"DEBUG (views.py): S3 upload confirmed for {pdf_doc.file.name} using storage.exists().")
        logger.info(
            f"S3 upload confirmed for {pdf_doc.file.name} using storage.exists().")

        print(
            f"DEBUG (views.py): Waiting for S3 object '{pdf_doc.file.name}' to be available via storage backend.")
        s3_available = wait_for_s3_file(pdf_doc.file.storage, pdf_doc.file.name)
        if not s3_available:
            print(
                f"ERROR (views.py): S3 object '{pdf_doc.file.name}' did not become available for download.")
            pdf_doc.delete()  # Clean up DB entry
            return JsonResponse({'status': 'error', 'message': 'Failed to confirm PDF availability in S3.'}, status=500)
        print(
            f"DEBUG (views.py): S3 object '{pdf_doc.file.name}' is now available.")

    try:
        start_total = time.time()
        print(
            f"DEBUG (views.py): Starting PDF processing for PDF ID: {pdf_doc.id}")

        print(
            f"DEBUG (views.py): Calling parse_pdf_ultra_fast for PDF ID: {pdf_doc.id}")
        pages_markdown = parse_pdf_ultra_fast(pdf_doc)
        pdf_doc.num_pages = len(pages_markdown)
        print(
            f"DEBUG (views.py): PDF parsed. Number of pages: {pdf_doc.num_pages}")

        print(f"DEBUG (views.py): Chunking documents for PDF ID: {pdf_doc.id}")
        docs = chunk_documents_ultra_fast(
            pages_markdown, pdf_doc.id, source=uploaded_file.name)
        pdf_doc.num_chunks = len(docs)
        print(
            f"DEBUG (views.py): Documents chunked. Number of chunks: {pdf_doc.num_chunks}")

        if settings.ENVIRONMENT == 'UAT_AWS':
            index_base_path = f"faiss_indexes/{pdf_doc.id}"
            print(f"DEBUG (views.py): Index base path (S3): {index_base_path}")
        else:
            index_base_path = os.path.join(
                settings.MEDIA_ROOT, 'faiss_indexes', str(pdf_doc.id)).replace(" ", "_")
            print(
                f"DEBUG (views.py): Index base path (Local): {index_base_path}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(build_bm25_fast, docs)
            vector_future = executor.submit(
                build_vector_store_fast, docs, pdf_doc.id, index_base_path)

            bm25_obj, tokenized_texts = bm25_future.result()
            vectorstore = vector_future.result()

        total_time = time.time() - start_total
        pdf_doc.processing_time = total_time
        pdf_doc.processed = True
        pdf_doc.save()
        logger.info(f"PDFDocument saved. File URL: {pdf_doc.file.url}")

        if not pages_markdown:
            raise ValueError("PDF parsing resulted in no content.")

        bm25_file_path = os.path.join(index_base_path, 'bm25.pkl')
        tokenized_texts_file_path = os.path.join(
            index_base_path, 'tokenized_texts.json')
        docs_file_path = os.path.join(index_base_path, 'docs.json')

        save_bm25_object(bm25_obj, pdf_doc.id, bm25_file_path)
        save_tokenized_texts(tokenized_texts, pdf_doc.id, tokenized_texts_file_path)
        save_docs(docs, pdf_doc.id, docs_file_path)

        pdf_doc.bm25_path = bm25_file_path
        pdf_doc.tokenized_texts_path = tokenized_texts_file_path
        pdf_doc.docs_path = docs_file_path
        pdf_doc.save()

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
        try:
            if settings.ENVIRONMENT != 'UAT_AWS':
                if 'pdf_doc' in locals() and getattr(pdf_doc, 'file', None):
                    try:
                        local_path = getattr(pdf_doc.file, 'path', None)
                        if local_path and os.path.exists(local_path):
                            os.remove(local_path)
                    except Exception:
                        pass
        except Exception:
            pass
        if 'pdf_doc' in locals() and pdf_doc.pk:
            pdf_doc.delete()
        return JsonResponse({'status': 'error', 'message': f'Error processing PDF: {e}'}, status=500)


@require_POST
@csrf_exempt
def chat(request):
    """Handles chat queries for a processed PDF."""
    pdf_id = request.POST.get('pdf_id')
    query = request.POST.get('query')

    if not pdf_id or not query:
        return JsonResponse({'status': 'error', 'message': 'Missing pdf_id or query.'}, status=400)

    try:
        pdf_doc = PDFDocument.objects.get(id=pdf_id)
        if not pdf_doc.processed:
            return JsonResponse({'status': 'error', 'message': 'PDF not yet processed.'}, status=400)

        data = processed_data_store.get(pdf_id)
        if not data:
            if pdf_doc.bm25_path and pdf_doc.tokenized_texts_path and pdf_doc.docs_path:
                try:
                    vectorstore = load_faiss_vector_store(
                        pdf_doc.id,
                        os.path.dirname(
                            pdf_doc.bm25_path) if settings.ENVIRONMENT != 'UAT_AWS' else ""
                    )
                    bm25_obj = load_bm25_object(pdf_doc.id, pdf_doc.bm25_path)
                    tokenized_texts = load_tokenized_texts(pdf_doc.id, pdf_doc.tokenized_texts_path)
                    docs = load_docs(pdf_doc.id, pdf_doc.docs_path)

                    if vectorstore is None or bm25_obj is None or not tokenized_texts or not docs:
                        raise ValueError(
                            "Failed to load all processed data components.")

                    processed_data_store[pdf_doc.id] = {
                        "docs": docs,
                        "vectorstore": vectorstore,
                        "bm25_obj": bm25_obj,
                        "tokenized_texts": tokenized_texts,
                    }
                except Exception as e:
                    logger.error(
                        f"Error loading processed data from disk/S3 for PDF {pdf_id}: {e}", exc_info=True)
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
        answer, chunk_data = answer_with_context_optimized(query, candidates)
        answer_time = time.time() - start_answer

        ChatHistory.objects.create(
            pdf_document=pdf_doc,
            question=query,
            answer=answer
        )

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
                "original_page": metadata.get('page', 'N/A')
            })

        return JsonResponse({
            'status': 'success',
            'answer': answer,
            'retrieval_time': f"{retrieval_time:.2f}s",
            'generation_time': f"{answer_time:.2f}s",
            'context': formatted_chunks
        })

    except PDFDocument.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'PDF document not found.'}, status=404)
    except Exception as e:
        logger.error(f"Error during chat: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'Error during chat: {e}'}, status=500)


@require_POST
@csrf_exempt
def summarize_chunk(request):
    """Generates a summary for a specific chunk of text."""
    pdf_id = request.POST.get('pdf_id')
    chunk_content = request.POST.get('chunk_content')
    chunk_page = request.POST.get('chunk_page')
    query = request.POST.get('query')

    logger.info(
        f"summarize_chunk received: pdf_id={pdf_id}, chunk_page={chunk_page}, query={query}, chunk_content_len={len(chunk_content) if chunk_content else 0}")

    if not pdf_id or not chunk_content or not chunk_page or not query:
        logger.error(
            "Missing pdf_id, chunk_content, chunk_page, or query for summarize_chunk.")
        return JsonResponse({'status': 'error', 'message': 'Missing pdf_id, chunk_content, chunk_page, or query.'}, status=400)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not configured for summarize_chunk.")
        return JsonResponse({'status': 'error', 'message': '❌ OpenAI API key not configured.'}, status=500)

    logger.info(
        f"OpenAI API key loaded for summarization: {'sk-proj-...' + openai_api_key[-5:] if openai_api_key else 'None'}")

    try:
        client = openai.OpenAI(api_key=openai_api_key)
        prompt = f"""Based EXCLUSIVELY on the following context, provide a concise and accurate summary of the text chunk in relation to the original question.
        
        Original Question: {query}
        
        Context Information (from Page: {chunk_page}):
        {chunk_content}
        
        Instructions:
        - Summarize the provided text chunk specifically answering or relating to the Original Question.
        - Use ONLY information from the provided Context Information.
        - If the context doesn't contain relevant information for the question, state \"This chunk does not contain information relevant to the question.\"
        - Be precise and keep the summary focused.
        - Keep the summary to a maximum of 100 words.
        
        Summary:"""
        logger.info(
            f"Summarization prompt for PDF ID {pdf_id}, Page {chunk_page}: {prompt[:500]}...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise assistant that summarizes text chunks based on a given question and context. Never hallucinate or use external knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200,
        )
        summary = response.choices[0].message.content
        logger.info(
            f"Successfully generated summary for PDF ID {pdf_id}, Page {chunk_page}. Summary: {summary[:200]}...")
        return JsonResponse({'status': 'success', 'summary': summary})

    except openai.APIStatusError as e:
        logger.error(
            f"OpenAI API Status Error during chunk summarization (PDF ID {pdf_id}, Page {chunk_page}): Status {e.status_code}, Response: {e.response}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'OpenAI API error: {e.status_code} - {e.message}'}, status=500)
    except openai.APIConnectionError as e:
        logger.error(
            f"OpenAI API Connection Error during chunk summarization (PDF ID {pdf_id}, Page {chunk_page}): {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'OpenAI API connection error: {e}'}, status=500)
    except openai.RateLimitError as e:
        logger.error(
            f"OpenAI Rate Limit Error during chunk summarization (PDF ID {pdf_id}, Page {chunk_page}): {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'OpenAI rate limit exceeded: {e}'}, status=429)
    except Exception as e:
        logger.error(
            f"Unexpected error during chunk summarization (PDF ID {pdf_id}, Page {chunk_page}): {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'Error generating summary: {e}'}, status=500)


@require_GET
def get_chat_history(request, pdf_id):
    """Retrieves chat history for a given PDF."""
    try:
        pdf_doc = PDFDocument.objects.get(id=pdf_id)
        history = ChatHistory.objects.filter(pdf_document=pdf_doc).order_by('timestamp')
        chat_entries = []
        for entry in history:
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
def clear_data(request):
    """Clears all processed data and PDF documents."""
    try:
        processed_data_store.clear()
        print("data cleared----------------------")

        pdf_docs_to_delete = list(PDFDocument.objects.all())
        print("pdf doc deleted-----------------------")
        s3_client = None
        if settings.ENVIRONMENT == 'UAT_AWS':
            s3_client = get_s3_client()
            s3_bucket = settings.AWS_STORAGE_BUCKET_NAME
            print("s3 bucket-----------------------")
            print(
                f"DEBUG (views.py): ENVIRONMENT is UAT_AWS. Initializing S3 client for clearing data from bucket: {s3_bucket}")

        for pdf_doc in pdf_docs_to_delete:
            print(f"DEBUG (views.py): Deleting data for PDF ID: {pdf_doc.id}")
            if pdf_doc.file:
                print(
                    f"DEBUG (views.py): Deleting PDF file {pdf_doc.file.name} from storage.")
                pdf_doc.file.delete(save=False)

            if settings.ENVIRONMENT == 'UAT_AWS' and s3_client:
                prefix = f"faiss_indexes/{pdf_doc.id}/"
                print(
                    f"DEBUG (views.py): Deleting S3 index objects under prefix: {prefix}")
                try:
                    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
                    if 'Contents' in response:
                        objects_to_delete = [{'Key': obj['Key']}
                                             for obj in response['Contents']]
                        if objects_to_delete:
                            s3_client.delete_objects(
                                Bucket=s3_bucket,
                                Delete={'Objects': objects_to_delete,
                                        'Quiet': True}
                            )
                            print(
                                f"DEBUG (views.py): Deleted {len(objects_to_delete)} S3 objects for PDF ID {pdf_doc.id} under prefix {prefix}")
                            logger.info(
                                f"Deleted S3 objects for PDF ID {pdf_doc.id} under prefix {prefix}")
                except Exception as e:
                    print(
                        f"ERROR (views.py): Error deleting S3 objects for PDF ID {pdf_doc.id}: {e}")
                    logger.error(
                        f"Error deleting S3 objects for PDF ID {pdf_doc.id}: {e}")
            else:
                index_dir_path = os.path.join(
                    settings.MEDIA_ROOT, 'faiss_indexes', str(pdf_doc.id))
                if os.path.isdir(index_dir_path):
                    import shutil
                    print(
                        f"DEBUG (views.py): Deleting local FAISS index directory: {index_dir_path}")
                    shutil.rmtree(index_dir_path)
                    logger.info(
                        f"Deleted local FAISS index directory: {index_dir_path}")

        print(f"DEBUG (views.py): Deleting all PDFDocument and ChatHistory entries from database.")
        PDFDocument.objects.all().delete()
        ChatHistory.objects.all().delete()

        if settings.ENVIRONMENT != 'UAT_AWS':
            media_root = settings.MEDIA_ROOT
            if os.path.exists(media_root) and not os.listdir(media_root):
                print(
                    f"DEBUG (views.py): Deleting empty local media root: {media_root}")
                os.rmdir(media_root)

            faiss_indexes_root = os.path.join(media_root, 'faiss_indexes')
            if os.path.exists(faiss_indexes_root) and not os.listdir(faiss_indexes_root):
                print(
                    f"DEBUG (views.py): Deleting empty local FAISS indexes root: {faiss_indexes_root}")
                os.rmdir(faiss_indexes_root)

        print(f"DEBUG (views.py): All data and cache cleared successfully.")
        return JsonResponse({'status': 'success', 'message': 'All data and cache cleared.'})
    except Exception as e:
        print(f"ERROR (views.py): Error clearing data: {e}")
        logger.error(f"Error clearing data: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'Error clearing data: {e}'}, status=500)
