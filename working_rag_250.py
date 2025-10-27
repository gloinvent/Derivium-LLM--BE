"""
Ultra-Fast PDF Chat: PyMuPDF + LangChain + FAISS
Optimized for 120-second parsing and embedding with advanced reranking
"""

import os
import tempfile
import json
from typing import List, Tuple, Dict, Any
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import time

import openai
from dotenv import load_dotenv
import streamlit as st

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
st.set_page_config(page_title="🚀 Ultra-Fast PDF Chat", layout="wide")

# Model and processing settings
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight reranker
CHUNK_SIZE = 1500  # Reduced for better precision
CHUNK_OVERLAP = 150
TOP_K = 10  # Retrieve more initially for reranking
TOP_K_FINAL = 3  # Final number after reranking

INDEX_DIR = "faiss_index_dir"

# OCR optimization
TESSERACT_CONFIG = r'--oem 3 --psm 3 -c preserve_interword_spaces=1 tessedit_do_invert=0'

# Parallel processing settings
MAX_WORKERS = min(8, os.cpu_count() or 4)
EMBEDDING_BATCH_SIZE = 100

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Model loaders with caching
# -------------------------
@st.cache_resource
def load_embedding_model():
    """Cache the embedding model to avoid reloading."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        show_progress_bar=False,
        request_timeout=60,
        chunk_size=EMBEDDING_BATCH_SIZE
    )

@st.cache_resource
def load_reranker_model():
    """Cache the reranker model for better performance."""
    return CrossEncoder(RERANKER_MODEL)

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

@st.cache_data(show_spinner=False, max_entries=1)
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
            st.warning("Document too large, using manual batching...")
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
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        status_text.text(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        progress_bar.progress(min((i + batch_size) / len(texts), 1.0))
        
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
    
    progress_bar.empty()
    status_text.empty()
    
    return vectorstore

@st.cache_resource
def build_vector_store_fast(_docs: List[Document], index_path: str = INDEX_DIR) -> FAISS:
    """Fast FAISS index building with batching."""
    return build_vector_store_with_batching(_docs, index_path)

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

def hybrid_retrieval_optimized(query: str, docs: List[Document],
                              vectorstore: FAISS, bm25_obj: BM25Okapi,
                              tokenized_texts: List[str], top_k: int = TOP_K) -> List[Document]:
    """Optimized hybrid retrieval with advanced reranking."""
    # Phase 1: Initial retrieval (more candidates)
    initial_k = min(top_k * 3, len(docs))
    
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
        for doc in docs:
            if (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk')) == key:
                scored_candidates.append((doc, score))
                break

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    initial_candidates = [doc for doc, _ in scored_candidates[:initial_k]]
    
    # Phase 2: Cross-encoder reranking for precision
    if len(initial_candidates) > TOP_K_FINAL:
        reranked_candidates = cross_encoder_reranking(query, initial_candidates)
        return reranked_candidates[:TOP_K_FINAL]
    
    return initial_candidates[:TOP_K_FINAL]

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

async def answer_with_context_optimized(query: str, candidates: List[Document]) -> str:
    """Optimized answer generation with better context assembly."""
    # Prioritize most relevant chunks
    context_parts = []
    for i, doc in enumerate(candidates[:4]):  # Use top 4 chunks max
        context_parts.append(
            f"[Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page {doc.metadata.get('page', 'N/A')}]:\n{doc.page_content}"
        )
    
    assembled_context = "\n\n".join(context_parts)

    prompt = f"""Based EXCLUSIVELY on the following context, provide a concise and accurate answer to the question.

Question: {query}

Context Information:
{assembled_context}

Instructions:
- Answer using ONLY information from the provided context
- If the context doesn't contain relevant information, state "I cannot find this information in the document"
- Be precise and cite the source pages when possible
- Keep the answer focused and avoid speculation

Answer:"""

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "❌ OpenAI API key not configured."

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
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error generating response: {e}"

# -------------------------
# Streamlit UI
# -------------------------

st.title("🚀 Ultra-Fast PDF Chat — Advanced Reranking")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tfile:
            tfile.write(uploaded_file.read())
            pdf_path = tfile.name

        st.success("✅ PDF uploaded successfully")

        if st.button("Process PDF", type="primary"):
            with st.spinner("🚀 Ultra-fast PDF Processing..."):
                start_total = time.time()
                
                # 1. Fast PDF parsing
                parse_start = time.time()
                pages_markdown = asyncio.run(parse_pdf_ultra_fast(pdf_path))
                parse_time = time.time() - parse_start
                st.info(f"📄 Parsed {len(pages_markdown)} pages in {parse_time:.2f}s")
                
                # 2. Fast chunking with deduplication
                chunk_start = time.time()
                docs = chunk_documents_ultra_fast(pages_markdown, source=uploaded_file.name)
                chunk_time = time.time() - chunk_start
                st.info(f"✂️ Created {len(docs)} unique chunks in {chunk_time:.2f}s")
                
                # 3. Parallel indexing
                index_start = time.time()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    bm25_future = executor.submit(build_bm25_fast, docs)
                    vector_future = executor.submit(build_vector_store_fast, docs)
                    
                    bm25_obj, tokenized_texts = bm25_future.result()
                    vectorstore = vector_future.result()
                
                index_time = time.time() - index_start
                st.info(f"🔍 Indexing completed in {index_time:.2f}s")
                
                total_time = time.time() - start_total
                
                st.session_state.update({
                    "docs": docs,
                    "vectorstore": vectorstore,
                    "bm25_obj": bm25_obj,
                    "tokenized_texts": tokenized_texts,
                    "processed": True
                })
                
                st.success(f"✅ Total processing time: {total_time:.2f}s")
                
                # Show performance breakdown
                with st.expander("Performance Details"):
                    st.metric("PDF Parsing", f"{parse_time:.2f}s")
                    st.metric("Chunking", f"{chunk_time:.2f}s") 
                    st.metric("Indexing", f"{index_time:.2f}s")
                    st.metric("Total", f"{total_time:.2f}s")

        if not st.session_state.get("processed") and os.path.isdir(INDEX_DIR):
            if st.button("Load Existing Index"):
                try:
                    vectorstore = FAISS.load_local(INDEX_DIR, load_embedding_model())
                    st.session_state["vectorstore"] = vectorstore
                    st.success("Loaded existing FAISS index")
                except Exception as e:
                    st.error(f"Failed to load vectorstore: {e}")

with col2:
    if st.session_state.get("processed"):
        query = st.text_input("Your question:", placeholder="Ask about the PDF content...")

        if st.button("Get Answer") and query.strip():
            with st.spinner("🔍 Advanced retrieval with reranking..."):
                start = time.time()
                candidates = hybrid_retrieval_optimized(
                    query,
                    st.session_state["docs"],
                    st.session_state["vectorstore"],
                    st.session_state["bm25_obj"],
                    st.session_state["tokenized_texts"]
                )
                retrieval_time = time.time() - start

                col_context, col_meta, col_answer = st.columns([2, 1, 2])
                
                with col_context:
                    st.subheader("📄 Retrieved Context (Reranked)")
                    for i, doc in enumerate(candidates):
                        with st.expander(f"Context {i+1} (Page {doc.metadata.get('page', 'N/A')})", expanded=i == 0):
                            st.write(doc.page_content)
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")

                with col_meta:
                    st.subheader("🔍 Retrieval Info")
                    st.metric("Retrieval Time", f"{retrieval_time:.2f}s")
                    st.metric("Total Chunks", len(candidates))
                    st.metric("Strategy", "Hybrid + Reranking")
                    
                    for i, doc in enumerate(candidates):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(f"Page: {doc.metadata.get('page', 'N/A')}")
                        st.write(f"Length: {doc.metadata.get('chunk_length', 0)} chars")

                with col_answer:
                    st.subheader("🤖 AI Response")
                    answer_start = time.time()
                    answer = asyncio.run(answer_with_context_optimized(query, candidates))
                    answer_time = time.time() - answer_start
                    
                    st.write(answer)
                    st.caption(f"Retrieval: {retrieval_time:.2f}s | Generation: {answer_time:.2f}s")

# -------------------------
# Sidebar management
# -------------------------
with st.sidebar:
    st.header("⚙️ Management")

    if st.button("Clear Cache & Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        if os.path.isdir(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared cache & session state")
        st.rerun()

    if st.session_state.get("docs"):
        st.download_button(
            "📥 Export Metadata",
            json.dumps([
                {
                    "source": d.metadata.get('source'),
                    "page": d.metadata.get('page'),
                    "chunk": d.metadata.get('chunk'),
                    "text_preview": d.page_content[:200],
                    "length": len(d.page_content)
                } for d in st.session_state["docs"][:100]
            ], indent=2),
            file_name="document_metadata.json",
            mime="application/json"
        )

    if st.sidebar.checkbox("Show Performance Info"):
        if st.session_state.get("docs"):
            st.sidebar.metric("Total Chunks", len(st.session_state["docs"]))
            avg_len = np.mean([len(d.page_content) for d in st.session_state["docs"]])
            st.sidebar.metric("Avg Chunk Length", f"{avg_len:.0f} chars")

# Cleanup temporary file
if 'pdf_path' in locals() and os.path.exists(pdf_path):
    os.unlink(pdf_path)