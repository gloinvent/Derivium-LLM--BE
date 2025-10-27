import os
import tempfile
import json
from typing import List, Tuple, Dict, Any
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import time
import pickle # Added for serializing BM25
import json # Added for serializing docs and tokenized_texts

import openai
from dotenv import load_dotenv

# PDF parsing - optimized imports
import pymupdf4llm
import fitz
from PIL import Image
import pytesseract

# LangChain helpers
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS as FAISS_CLASS
from pathlib import Path # Added for path manipulation

# Sparse retrieval and reranking
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder

# Utilities
import faiss
import cv2

# -------------------------
# Optimized Configuration
# -------------------------
load_dotenv()

# Model and processing settings
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight reranker
CHUNK_SIZE = 1500  # Reduced for better precision
CHUNK_OVERLAP = 150
TOP_K = 10  # Retrieve more initially for reranking
TOP_K_FINAL = 3  # Final number after reranking

INDEX_DIR = "faiss_index_dir" # This will need to be adjusted for Django's MEDIA_ROOT

# OCR optimization
TESSERACT_CONFIG = r'--oem 3 --psm 3 -c preserve_interword_spaces=1 tessedit_do_invert=0'

# Parallel processing settings
MAX_WORKERS = min(8, os.cpu_count() or 4)
EMBEDDING_BATCH_SIZE = 100

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Model loaders (without Streamlit caching)
# -------------------------
_embedding_model_instance = None
_reranker_model_instance = None

def load_embedding_model():
    """Load the embedding model."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            show_progress_bar=False,
            request_timeout=60,
            chunk_size=EMBEDDING_BATCH_SIZE
        )
    return _embedding_model_instance

def load_reranker_model():
    """Load the reranker model."""
    global _reranker_model_instance
    if _reranker_model_instance is None:
        _reranker_model_instance = CrossEncoder(RERANKER_MODEL)
    return _reranker_model_instance

# -------------------------
# Ultra-Fast PDF Parsing
# -------------------------

def extract_text_optimized(page: fitz.Page) -> Tuple[str, bool]:
    """
    Ultra-fast text extraction with priority:
    1. PyMuPDF4LLM markdown
    2. Native PyMuPDF text
    3. Fast OCR fallback (only if absolutely necessary)
    """
    # Try pymupdf4llm first (fastest)
    try:
        temp_doc = fitz.open()
        temp_doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
        markdown_text = pymupdf4llm.to_markdown(temp_doc)
        temp_doc.close()

        if len(markdown_text.strip()) > 50:
            return markdown_text, True
    except Exception as e:
        logger.debug(f"pymupdf4llm failed for page {page.number}: {e}")

    # Try native text extraction
    native_text = page.get_text("text", sort=True, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_IMAGES)
    if len(native_text.strip()) > 50:
        return native_text, True

    # Quick OCR check - only if page has images
    image_list = page.get_images()
    if image_list:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Fast OCR with minimal preprocessing
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_text = pytesseract.image_to_string(binary, config=TESSERACT_CONFIG)

            if len(ocr_text.strip()) > 30:
                return ocr_text, False
        except Exception as e:
            logger.debug(f"OCR failed for page {page.number}: {e}")

    return "", False

def process_page_batch(args: Tuple[str, List[int]]) -> List[Tuple[int, str]]:
    """Process multiple pages in batch for reduced overhead."""
    file_path, page_indices = args
    results = []

    with fitz.open(file_path) as doc:
        for page_num in page_indices:
            try:
                page = doc[page_num]
                text, is_high_quality = extract_text_optimized(page)
                if text:
                    results.append((page_num, text))
            except Exception as e:
                logger.debug(f"Page {page_num} failed: {e}")
                continue

    return results

async def parse_pdf_ultra_fast(file_path: str) -> List[Tuple[int, str]]:
    """Ultra-fast PDF parsing with optimal parallelization."""
    start_time = time.time()

    with fitz.open(file_path) as doc:
        total_pages = doc.page_count

        # Create page batches for parallel processing
        page_batches = []
        batch_size = max(1, total_pages // MAX_WORKERS)

        for i in range(0, total_pages, batch_size):
            page_batches.append((file_path, list(range(i, min(i + batch_size, total_pages)))))

        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, process_page_batch, batch)
                for batch in page_batches
            ]
            batch_results = await asyncio.gather(*tasks)

        # Flatten results
        all_results = []
        for batch in batch_results:
            all_results.extend(batch)

        # Sort by page number
        all_results.sort(key=lambda x: x[0])

        parsing_time = time.time() - start_time
        logger.info(f"PDF parsed {total_pages} pages in {parsing_time:.2f}s")

        return all_results

# -------------------------
# Optimized Document Processing
# -------------------------

def chunk_documents_ultra_fast(pages_markdown: List[Tuple[int, str]], source: str = "uploaded.pdf") -> List[Document]:
    """Ultra-fast document chunking with minimal overhead."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    docs = []
    for page_num, page_md_text in pages_markdown:
        # Fast text cleaning
        clean_text = " ".join(page_md_text.split())
        if len(clean_text.strip()) < 25:
            continue

        chunks = splitter.split_text(clean_text)
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            metadata = {
                "source": source,
                "page": page_num + 1,
                "chunk": i,
                "chunk_length": len(chunk),
                "text_hash": hash(chunk)  # For deduplication
            }
            docs.append(Document(page_content=chunk, metadata=metadata))

    # Remove duplicates based on content hash
    unique_docs = []
    seen_hashes = set()
    for doc in docs:
        text_hash = doc.metadata["text_hash"]
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_docs.append(doc)

    return unique_docs

def build_vector_store_with_batching(_docs: List[Document], index_path: str = INDEX_DIR) -> FAISS:
    """FAISS index building with proper batching to avoid token limits."""
    embeddings = load_embedding_model()

    try:
        vectorstore = FAISS.from_documents(_docs, embeddings)
    except Exception as e:
        if "max_tokens_per_request" in str(e) or "400" in str(e):
            logger.warning("Document too large, using manual batching...")
            vectorstore = _build_faiss_manual_batching(_docs, embeddings, index_path)
        else:
            raise e

    # Save index
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)

    return vectorstore

def _build_faiss_manual_batching(docs: List[Document], embeddings, index_path: str) -> FAISS:
    """Manual batching for very large documents."""
    from langchain_community.vectorstores.utils import DistanceStrategy

    texts = [doc.page_content for doc in docs]

    # Batch process embeddings
    all_embeddings = []
    batch_size = EMBEDDING_BATCH_SIZE

    # No Streamlit progress bar here, just log
    logger.info(f"Starting manual batch embedding for {len(texts)} documents.")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        try:
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.warning(f"Batch {i} failed, reducing batch size: {e}")
            # Try with smaller batch
            smaller_batch = batch_size // 2
            for j in range(0, len(batch_texts), smaller_batch):
                small_batch_texts = batch_texts[j:j + smaller_batch]
                try:
                    small_embeddings = embeddings.embed_documents(small_batch_texts)
                    all_embeddings.extend(small_embeddings)
                except Exception as e2:
                    logger.error(f"Small batch also failed: {e2}")
                    continue

    # Create FAISS vector store with manual embedding
    embedding_matrix = np.array(all_embeddings).astype('float32')

    # Create index
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=FAISS._build_docstore(docs),
        index_to_docstore_id=FAISS._build_index_to_docstore_id(docs),
        distance_strategy=DistanceStrategy.COSINE
    )

    return vectorstore

def build_vector_store_fast(_docs: List[Document], index_path: str = INDEX_DIR) -> FAISS_CLASS:
    """Fast FAISS index building with batching."""
    return build_vector_store_with_batching(_docs, index_path)

def load_faiss_vector_store(index_path: str) -> FAISS_CLASS:
    """Load FAISS index with dangerous deserialization allowed for trusted sources."""
    embeddings = load_embedding_model()
    # Resolve the path to handle any potential issues with spaces or relative paths
    resolved_index_path = Path(index_path).resolve()
    return FAISS_CLASS.load_local(str(resolved_index_path), embeddings, allow_dangerous_deserialization=True)

def save_bm25_object(bm25_obj: BM25Okapi, path: str):
    """Saves a BM25Okapi object to disk using pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(bm25_obj, f)

def load_bm25_object(path: str) -> BM25Okapi:
    """Loads a BM25Okapi object from disk using pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_tokenized_texts(tokenized_texts: List[str], path: str):
    """Saves tokenized texts to disk as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tokenized_texts, f, ensure_ascii=False, indent=2)

def load_tokenized_texts(path: str) -> List[str]:
    """Loads tokenized texts from disk as JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_docs(docs: List[Document], path: str):
    """Saves a list of Document objects to disk as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable_docs = [{
        "page_content": doc.page_content,
        "metadata": doc.metadata
    } for doc in docs]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

def load_docs(path: str) -> List[Document]:
    """Loads a list of Document objects from disk (JSON)."""
    with open(path, 'r', encoding='utf-8') as f:
        serializable_docs = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in serializable_docs]

def build_bm25_fast(docs: List[Document]) -> Tuple[BM25Okapi, List[str]]:
    """Optimized BM25 corpus building."""
    tokenized_corpus = []
    clean_texts = []

    for doc in docs:
        tokens = doc.page_content.lower().split()
        if len(tokens) > 3:
            tokenized_corpus.append(tokens)
            clean_texts.append(" ".join(tokens))

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, clean_texts

# -------------------------
# Advanced Hybrid Retrieval with Reranking
# -------------------------

def retrieve_chunks(query: str, docs: List[Document],
                    vectorstore: FAISS, bm25_obj: BM25Okapi,
                    top_k: int = TOP_K) -> List[Document]:
    """
    Performs optimized hybrid retrieval (dense + sparse) and prepares candidates for reranking.
    """
    initial_k = min(top_k * 3, len(docs))

    # Create a dictionary for O(1) lookup of documents by their key
    doc_map = {}
    for doc in docs:
        key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk'))
        doc_map[key] = doc

    # Dense retrieval
    try:
        dense_docs = vectorstore.similarity_search_with_score(query, k=initial_k)
        dense_results, dense_scores = zip(*dense_docs) if dense_docs else ([], [])
    except Exception as e:
        logger.warning(f"Dense retrieval failed: {e}")
        dense_results, dense_scores = [], []

    # Sparse retrieval
    query_tokens = query.lower().split()
    bm25_scores = bm25_obj.get_scores(query_tokens)

    # Normalize BM25 scores
    if len(bm25_scores) > 0:
        max_bm25 = np.max(bm25_scores)
        bm25_scores = bm25_scores / max_bm25 if max_bm25 > 0 else np.zeros_like(bm25_scores)

    # Combine initial scores
    candidate_scores = {}

    # Add dense results (weighted higher for semantic understanding)
    for doc, score in zip(dense_results, dense_scores):
        key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk'))
        dense_norm_score = 1 - (score / 2) if score <= 2 else 0
        candidate_scores[key] = candidate_scores.get(key, 0) + 0.6 * dense_norm_score

    # Add sparse results (weighted for keyword matching)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:initial_k]
    for idx in top_bm25_indices:
        if idx < len(docs):
            doc = docs[idx]
            key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk'))
            candidate_scores[key] = candidate_scores.get(key, 0) + 0.4 * bm25_scores[idx]

    # Get top candidates for reranking
    scored_candidates = []
    for key, score in candidate_scores.items():
        doc = doc_map.get(key)
        if doc:
            scored_candidates.append((doc, score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_candidates[:initial_k]]


def hybrid_retrieval_optimized(query: str, docs: List[Document],
                               vectorstore: FAISS, bm25_obj: BM25Okapi,
                               tokenized_texts: List[str], top_k: int = TOP_K) -> List[Document]:
    """Optimized hybrid retrieval with advanced reranking and expanded page context."""
    # Phase 1: Initial retrieval (more candidates)
    initial_candidates = retrieve_chunks(query, docs, vectorstore, bm25_obj, top_k)

    # Phase 2: Cross-encoder reranking for precision
    if len(initial_candidates) > TOP_K_FINAL:
        reranked_candidates = cross_encoder_reranking(query, initial_candidates)
        top_reranked_chunks = reranked_candidates[:TOP_K_FINAL]
    else:
        top_reranked_chunks = initial_candidates[:TOP_K_FINAL]

    # Phase 3: Expand context by combining chunks from the relevant page and its neighbors into single documents
    final_context_docs = []
    processed_page_ranges = set() # To keep track of page ranges already processed

    # Group all original documents by page number for easy lookup
    docs_by_page: Dict[int, List[Document]] = {}
    for doc in docs:
        page_num = doc.metadata.get('page')
        if page_num is not None:
            if page_num not in docs_by_page:
                docs_by_page[page_num] = []
            docs_by_page[page_num].append(doc)

    for doc in top_reranked_chunks:
        main_page_num = doc.metadata.get('page')
        if main_page_num is None:
            continue

        # Determine the page range to combine (main_page, main_page-1, main_page+1)
        pages_to_combine = sorted(list(set([
            p for p in [main_page_num - 1, main_page_num, main_page_num + 1] if p > 0 and p in docs_by_page
        ])))

        if not pages_to_combine:
            continue

        # Create a unique identifier for this combined page range
        page_range_key = tuple(pages_to_combine)
        if page_range_key in processed_page_ranges:
            continue # Skip if this range has already been processed

        combined_content_parts = []
        combined_metadata_pages = []
        
        for p_num in pages_to_combine:
            if p_num in docs_by_page:
                # Sort chunks within the page to maintain order
                sorted_chunks_on_page = sorted(docs_by_page[p_num], key=lambda x: x.metadata.get('chunk', 0))
                for chunk_doc in sorted_chunks_on_page:
                    combined_content_parts.append(chunk_doc.page_content)
                combined_metadata_pages.append(p_num)

        if combined_content_parts:
            # Create a new Document representing the combined context
            # Use the metadata from the primary chunk, but update page info
            new_metadata = doc.metadata.copy()
            new_metadata['page'] = main_page_num # Keep the original page for primary reference
            new_metadata['pages'] = sorted(list(set(combined_metadata_pages))) # Store all pages included
            new_metadata['combined_context'] = True # Flag for easier identification

            combined_page_content = "\n\n".join(combined_content_parts)
            final_context_docs.append(Document(page_content=combined_page_content, metadata=new_metadata))
            processed_page_ranges.add(page_range_key)

    # Sort the final documents by their primary page number for consistent output
    final_context_docs.sort(key=lambda x: x.metadata.get('page', 0))

    return final_context_docs

def cross_encoder_reranking(query: str, candidates: List[Document]) -> List[Document]:
    """Use cross-encoder for precise reranking of top candidates."""
    try:
        reranker = load_reranker_model()

        # Prepare pairs for reranking
        pairs = [(query, doc.page_content) for doc in candidates]

        # Get reranker scores
        scores = reranker.predict(pairs)

        # Sort candidates by reranker scores
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs]

    except Exception as e:
        logger.warning(f"Reranking failed: {e}, falling back to original ranking")
        return candidates

# -------------------------
# LLM Answer Generation
# -------------------------

async def answer_with_context_optimized(query: str, candidates: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    """Optimized answer generation with better context assembly, handling expanded page context."""
    context_parts = []
    
    # Group candidates by page number for better context presentation
    pages_to_chunks: Dict[int, List[Document]] = {}
    for doc in candidates:
        page_num = doc.metadata.get('page')
        if page_num not in pages_to_chunks:
            pages_to_chunks[page_num] = []
        pages_to_chunks[page_num].append(doc)

    # Sort pages and then chunks within each page
    sorted_page_nums = sorted(pages_to_chunks.keys())

    for page_num in sorted_page_nums:
        chunks_on_page = sorted(pages_to_chunks[page_num], key=lambda x: x.metadata.get('chunk', 0))
        
        source_info = chunks_on_page[0].metadata.get('source', 'Unknown')
        
        # Combine all chunk content for the current page
        page_content = "\n".join([doc.page_content for doc in chunks_on_page])
        
        context_parts.append(
            f"[Source: {source_info}, Page: {page_num}]:\n{page_content}"
        )

    assembled_context = "\n\n".join(context_parts)

    prompt = f"""Based EXCLUSIVELY on the following context, provide a concise and accurate answer to the question.

Question: {query}

Context Information:
{assembled_context}

Instructions:
- Answer using ONLY information from the provided context
- If the context doesn't contain relevant information, state "I cannot find this information in the document"
- Be precise and cite the source pages when possible and give page numbers correctly if multiple pages are referenced
- Keep the answer focused and avoid speculation

Answer:"""

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "❌ OpenAI API key not configured.", []

    try:
        client = openai.AsyncOpenAI(api_key=openai_api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise assistant that answers questions based strictly on provided context. Never hallucinate or use external knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800,
        )
        
        # Prepare chunks for return
        chunk_data = []
        for doc in candidates:
            chunk_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return response.choices[0].message.content, chunk_data
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error generating response: {e}", []