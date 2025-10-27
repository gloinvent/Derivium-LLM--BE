# """
# Ultra-Fast PDF Chat: PyMuPDF + LangChain + FAISS
# Updated: include previous & next page into the context for top candidates,
# ensure ordering is prev -> candidate -> next (e.g. pages 1,2,3) in the combined chunk,
# rerank expanded contexts, then generate using the best expanded chunk.
# """

# import os
# import tempfile
# import json
# from typing import List, Tuple, Dict, Any
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# import logging
# import time

# import openai
# from dotenv import load_dotenv
# import streamlit as st

# # PDF parsing - optimized imports
# import pymupdf4llm
# import fitz
# from PIL import Image
# import pytesseract

# # LangChain helpers
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS

# # Sparse retrieval and reranking
# from rank_bm25 import BM25Okapi
# import numpy as np
# from sentence_transformers import CrossEncoder

# # Utilities
# import faiss
# import cv2
# import threading

# # -------------------------
# # Optimized Configuration
# # -------------------------
# load_dotenv()
# st.set_page_config(page_title="🚀 Ultra-Fast PDF Chat", layout="wide")

# # Model and processing settings
# EMBEDDING_MODEL = "text-embedding-3-small"
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight reranker
# CHUNK_SIZE = 2000  # Reduced for better precision
# CHUNK_OVERLAP = 200
# TOP_K = 10  # Retrieve more initially for reranking
# TOP_K_FINAL = 3  # Final number after reranking

# INDEX_DIR = "faiss_index_dir"

# # OCR optimization
# TESSERACT_CONFIG = r'--oem 3 --psm 3 -c preserve_interword_spaces=1 tessedit_do_invert=0'

# # Parallel processing settings
# MAX_WORKERS = min(8, os.cpu_count() or 4)
# EMBEDDING_BATCH_SIZE = 64  # somewhat smaller to avoid throttling in some embeddings

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # -------------------------
# # Model loaders with caching
# # -------------------------
# @st.cache_resource
# def load_embedding_model():
#     """Cache the embedding model to avoid reloading."""
#     return OpenAIEmbeddings(
#         model=EMBEDDING_MODEL,
#         show_progress_bar=False,
#         request_timeout=60,
#         chunk_size=EMBEDDING_BATCH_SIZE
#     )

# @st.cache_resource
# def load_reranker_model():
#     """Cache the reranker model for better performance."""
#     return CrossEncoder(RERANKER_MODEL)

# # -------------------------
# # Ultra-Fast PDF Parsing
# # -------------------------

# def extract_text_optimized(page: fitz.Page) -> Tuple[str, bool]:
#     """
#     Ultra-fast text extraction with priority:
#     1. pymupdf4llm markdown attempt (fallback if rich text)
#     2. Native PyMuPDF text (fast)
#     3. Fast OCR fallback (only if necessary)
#     """
#     # 1) Try pymupdf4llm markdown for richer formatting (kept as fallback)
#     try:
#         temp_doc = fitz.open()
#         try:
#             temp_doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
#             markdown_text = pymupdf4llm.to_markdown(temp_doc)
#             if markdown_text and len(markdown_text.strip()) > 50:
#                 return markdown_text, True
#         finally:
#             temp_doc.close()
#     except Exception as e:
#         logger.debug(f"pymupdf4llm failed for page {page.number}: {e}")

#     # 2) Try native text extraction (fast, low overhead)
#     try:
#         native_text = page.get_text("text", sort=True)
#         if native_text:
#             native_text_clean = native_text.replace("-\n", "")
#             if len(native_text_clean.strip()) > 50:
#                 return native_text_clean, True
#     except Exception as e:
#         logger.debug(f"Native text extraction failed for page {page.number}: {e}")

#     # 3) Quick OCR check - only if page has images and no text found
#     try:
#         image_list = page.get_images()
#         if image_list:
#             pix = page.get_pixmap(matrix=fitz.Matrix(100/72, 100/72))
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             arr = np.array(img)
#             if arr.ndim == 3:
#                 gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
#             else:
#                 gray = arr
#             _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             ocr_text = pytesseract.image_to_string(binary, config=TESSERACT_CONFIG)
#             if ocr_text and len(ocr_text.strip()) > 30:
#                 return ocr_text, False
#     except Exception as e:
#         logger.debug(f"OCR failed for page {page.number}: {e}")

#     return "", False

# def process_page_batch(args: Tuple[str, List[int]]) -> List[Tuple[int, str]]:
#     """Process multiple pages in batch for reduced overhead (thread worker)."""
#     file_path, page_indices = args
#     results = []
#     try:
#         with fitz.open(file_path) as doc:
#             for page_num in page_indices:
#                 try:
#                     page = doc[page_num]
#                     text, _ = extract_text_optimized(page)
#                     if text:
#                         results.append((page_num, text))
#                 except Exception as e:
#                     logger.debug(f"Page {page_num} failed: {e}")
#                     continue
#     except Exception as e:
#         logger.debug(f"Failed opening doc in worker for {file_path}: {e}")
#     return results

# async def parse_pdf_ultra_fast(file_path: str) -> List[Tuple[int, str]]:
#     """Ultra-fast PDF parsing with optimized thread parallelization."""
#     start_time = time.time()
#     with fitz.open(file_path) as doc:
#         total_pages = doc.page_count

#     page_indices_all = list(range(total_pages))
#     batch_count = max(1, min(MAX_WORKERS, total_pages))
#     per_batch = (total_pages + batch_count - 1) // batch_count
#     page_batches = []
#     for i in range(0, total_pages, per_batch):
#         page_batches.append((file_path, list(range(i, min(i + per_batch, total_pages)))))

#     loop = asyncio.get_running_loop()
#     results = []
#     with ThreadPoolExecutor(max_workers=batch_count) as executor:
#         tasks = [loop.run_in_executor(executor, process_page_batch, batch) for batch in page_batches]
#         batch_results = await asyncio.gather(*tasks)
#         for batch in batch_results:
#             if batch:
#                 results.extend(batch)

#     results.sort(key=lambda x: x[0])
#     parsing_time = time.time() - start_time
#     logger.info(f"PDF parsed {total_pages} pages in {parsing_time:.2f}s")
#     return results

# # -------------------------
# # Optimized Document Processing
# # -------------------------

# @st.cache_data(show_spinner=False, max_entries=1)
# def chunk_documents_ultra_fast(pages_markdown: List[Tuple[int, str]], source: str = "uploaded.pdf") -> List[Document]:
#     """Ultra-fast document chunking with minimal overhead."""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
#     )

#     docs: List[Document] = []
#     for page_num, page_md_text in pages_markdown:
#         clean_text = " ".join(page_md_text.split())
#         if len(clean_text.strip()) < 40:
#             continue

#         chunks = splitter.split_text(clean_text)
#         for i, chunk in enumerate(chunks):
#             if len(chunk.strip()) < 50:
#                 continue
#             metadata = {
#                 "source": source,
#                 "page": page_num + 1,  # 1-indexed pages
#                 "chunk": i,
#                 "chunk_length": len(chunk),
#                 "text_hash": hash(chunk)
#             }
#             docs.append(Document(page_content=chunk, metadata=metadata))

#     # deduplicate
#     unique_docs = []
#     seen_hashes = set()
#     for doc in docs:
#         h = doc.metadata["text_hash"]
#         if h not in seen_hashes:
#             seen_hashes.add(h)
#             unique_docs.append(doc)

#     return unique_docs

# def _save_faiss_in_thread(vectorstore: FAISS, index_path: str):
#     """Save FAISS in separate thread to keep UI responsive."""
#     try:
#         os.makedirs(index_path, exist_ok=True)
#         vectorstore.save_local(index_path)
#         logger.info("FAISS saved to disk")
#     except Exception as e:
#         logger.warning(f"Failed to save FAISS index: {e}")

# def build_vector_store_with_batching(_docs: List[Document], index_path: str = INDEX_DIR) -> FAISS:
#     """FAISS index building with proper batching to avoid token limits."""
#     embeddings = load_embedding_model()

#     try:
#         vectorstore = FAISS.from_documents(_docs, embeddings)
#     except Exception as e:
#         if "max_tokens_per_request" in str(e) or "400" in str(e):
#             st.warning("Document too large, using manual batching...")
#             vectorstore = _build_faiss_manual_batching(_docs, embeddings, index_path)
#         else:
#             raise e

#     saver = threading.Thread(target=_save_faiss_in_thread, args=(vectorstore, index_path), daemon=True)
#     saver.start()

#     return vectorstore

# def _build_faiss_manual_batching(docs: List[Document], embeddings, index_path: str) -> FAISS:
#     """Manual batching for very large documents."""
#     from langchain_community.vectorstores.utils import DistanceStrategy

#     texts = [doc.page_content for doc in docs]

#     all_embeddings = []
#     batch_size = EMBEDDING_BATCH_SIZE

#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     try:
#         total = len(texts)
#         for i in range(0, total, batch_size):
#             batch_texts = texts[i:i + batch_size]
#             status_text.text(f"Embedding batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
#             progress_bar.progress(min((i + batch_size) / total, 1.0))
#             try:
#                 batch_embeddings = embeddings.embed_documents(batch_texts)
#                 all_embeddings.extend(batch_embeddings)
#             except Exception as e:
#                 logger.warning(f"Batch {i} failed, reducing batch size: {e}")
#                 smaller_batch = max(1, batch_size // 2)
#                 for j in range(0, len(batch_texts), smaller_batch):
#                     small_batch_texts = batch_texts[j:j + smaller_batch]
#                     try:
#                         small_embeddings = embeddings.embed_documents(small_batch_texts)
#                         all_embeddings.extend(small_embeddings)
#                     except Exception as e2:
#                         logger.error(f"Small batch also failed: {e2}")
#                         continue

#         embedding_matrix = np.array(all_embeddings).astype('float32')
#         index = faiss.IndexFlatIP(embedding_matrix.shape[1])
#         index.add(embedding_matrix)

#         vectorstore = FAISS(
#             embedding_function=embeddings,
#             index=index,
#             docstore=FAISS._build_docstore(docs),
#             index_to_docstore_id=FAISS._build_index_to_docstore_id(docs),
#             distance_strategy=DistanceStrategy.COSINE
#         )
#     finally:
#         progress_bar.empty()
#         status_text.empty()

#     saver = threading.Thread(target=_save_faiss_in_thread, args=(vectorstore, index_path), daemon=True)
#     saver.start()

#     return vectorstore

# @st.cache_resource
# def build_vector_store_fast(_docs: List[Document], index_path: str = INDEX_DIR) -> FAISS:
#     """Fast FAISS index building with batching (cached)."""
#     return build_vector_store_with_batching(_docs, index_path)

# def build_bm25_fast(docs: List[Document]) -> Tuple[BM25Okapi, List[str]]:
#     """Optimized BM25 corpus building (faster tokenization)."""
#     tokenized_corpus = []
#     clean_texts = []
#     for doc in docs:
#         txt = doc.page_content.strip().lower()
#         tokens = txt.split()
#         if len(tokens) > 3:
#             tokenized_corpus.append(tokens)
#             clean_texts.append(" ".join(tokens))
#     bm25 = BM25Okapi(tokenized_corpus)
#     return bm25, clean_texts

# # -------------------------
# # Helper: Expand candidate with prev/next pages (preserve order prev -> candidate -> next)
# # -------------------------
# def expand_with_neighbor_pages(candidate: Document, all_docs: List[Document], neighbor_pages: int = 1) -> Document:
#     """
#     Given a candidate doc (one chunk), create a new Document which contains:
#     - chunks from previous pages (ascending order)
#     - the candidate chunk (in the middle)
#     - chunks from next pages (ascending order)
#     Example: if candidate page is 2 and neighbor_pages=1 -> order: page1, page2, page3
#     """
#     try:
#         src = candidate.metadata.get("source")
#         page = candidate.metadata.get("page", None)
#         if page is None:
#             return candidate  # nothing to expand

#         # compute pages to include
#         start_page = max(1, page - neighbor_pages)
#         end_page = page + neighbor_pages
#         pages_to_include = list(range(start_page, end_page + 1))

#         # collect chunks by page (preserve chunk order within a page)
#         page_to_chunks: Dict[int, List[Tuple[int, str]]] = {}
#         for doc in all_docs:
#             if doc.metadata.get("source") != src:
#                 continue
#             doc_page = doc.metadata.get("page", None)
#             if doc_page in pages_to_include:
#                 chunk_idx = doc.metadata.get("chunk", 0)
#                 page_to_chunks.setdefault(doc_page, []).append((chunk_idx, doc.page_content))

#         # sort chunks within each page by chunk index
#         for p in list(page_to_chunks.keys()):
#             page_to_chunks[p].sort(key=lambda x: x[0])

#         # Build ordered combined_text: previous pages in ascending order, candidate page chunks (with candidate chunk placed in its natural place), then next pages
#         ordered_blocks: List[str] = []
#         included_pages = set()

#         for p in pages_to_include:
#             chunks = page_to_chunks.get(p, [])
#             if not chunks:
#                 continue
#             # for each chunk in page, append with header
#             for idx, content in chunks:
#                 header = f"[Page {p} Chunk {idx}]"
#                 ordered_blocks.append(f"{header}\n{content}")
#             included_pages.add(p)

#         # Ensure candidate chunk marker is present and in correct relative position:
#         # The above loop already adds candidate page chunks in ascending order; nothing else needed.
#         combined = "\n\n".join(ordered_blocks)

#         new_meta = dict(candidate.metadata)
#         new_meta["expanded_pages"] = sorted(list(included_pages))
#         new_meta["expanded"] = True
#         new_meta["orig_page"] = page
#         new_meta["combined_chunk_length"] = len(combined)

#         return Document(page_content=combined, metadata=new_meta)
#     except Exception as e:
#         logger.debug(f"Failed to expand candidate: {e}")
#         return candidate

# # -------------------------
# # Advanced Hybrid Retrieval with Reranking
# # -------------------------

# def cross_encoder_reranking(query: str, candidates: List[Document]) -> List[Document]:
#     """Use cross-encoder for precise reranking of top candidates. Batches predictions for speed."""
#     try:
#         if not candidates:
#             return []

#         reranker = load_reranker_model()
#         pairs = [(query, doc.page_content) for doc in candidates]

#         batch_size = 16
#         scores = []
#         for i in range(0, len(pairs), batch_size):
#             batch_pairs = pairs[i:i + batch_size]
#             batch_scores = reranker.predict(batch_pairs)
#             scores.extend(batch_scores)

#         scored_docs = list(zip(candidates, scores))
#         scored_docs.sort(key=lambda x: x[1], reverse=True)
#         return [doc for doc, _ in scored_docs]
#     except Exception as e:
#         logger.warning(f"Reranking failed: {e}, falling back to original ranking")
#         return candidates

# def hybrid_retrieval_optimized(query: str, docs: List[Document],
#                               vectorstore: FAISS, bm25_obj: BM25Okapi,
#                               tokenized_texts: List[str], top_k: int = TOP_K) -> List[Document]:
#     """Optimized hybrid retrieval with advanced reranking + neighbor expansion."""
#     # Phase 1: Initial retrieval (more candidates)
#     initial_k = min(top_k * 3, len(docs))

#     # Dense retrieval
#     dense_results = []
#     dense_scores = []
#     try:
#         dense_docs_with_scores = vectorstore.similarity_search_with_score(query, k=initial_k)
#         if dense_docs_with_scores:
#             dense_results, dense_scores = zip(*dense_docs_with_scores)
#         else:
#             dense_results, dense_scores = [], []
#     except Exception as e:
#         logger.warning(f"Dense retrieval failed: {e}")
#         dense_results, dense_scores = [], []

#     # Sparse retrieval using BM25
#     query_tokens = query.lower().split()
#     if len(query_tokens) == 0:
#         bm25_scores = np.zeros(len(tokenized_texts))
#     else:
#         bm25_scores = bm25_obj.get_scores(query_tokens)

#     # Normalize BM25 scores safely
#     if len(bm25_scores) > 0:
#         max_bm25 = np.max(bm25_scores)
#         bm25_scores = bm25_scores / max_bm25 if max_bm25 > 0 else np.zeros_like(bm25_scores)

#     # Combine initial scores (use small dictionary keyed by doc tuple)
#     candidate_scores = {}

#     # add dense results (weight semantic higher)
#     for doc, score in zip(dense_results, dense_scores):
#         key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk'))
#         try:
#             dense_norm_score = float(score)
#             dense_norm_score = 1 / (1 + np.exp(-dense_norm_score))
#         except Exception:
#             dense_norm_score = 0.5
#         candidate_scores[key] = candidate_scores.get(key, 0) + 0.65 * dense_norm_score

#     # add sparse results (keyword matching)
#     if len(bm25_scores) > 0:
#         top_bm25_indices = np.argsort(bm25_scores)[::-1][:initial_k]
#         for idx in top_bm25_indices:
#             if idx < len(docs):
#                 doc = docs[idx]
#                 key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk'))
#                 candidate_scores[key] = candidate_scores.get(key, 0) + 0.35 * float(bm25_scores[idx])

#     # convert candidate_scores back to document list (keep top initial_k)
#     scored_candidates = []
#     for key, score in candidate_scores.items():
#         for doc in docs:
#             if (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk')) == key:
#                 scored_candidates.append((doc, score))
#                 break

#     scored_candidates.sort(key=lambda x: x[1], reverse=True)
#     initial_candidates = [doc for doc, _ in scored_candidates[:initial_k]]

#     # Phase 2: Cross-encoder reranking for precision (first pass)
#     max_rerank = max(10, TOP_K_FINAL * 4)
#     rerank_slice = initial_candidates[:max_rerank]

#     if len(rerank_slice) > 1:
#         reranked_candidates = cross_encoder_reranking(query, rerank_slice)
#     else:
#         reranked_candidates = rerank_slice

#     # Keep top-K_FINAL from first reranking
#     top_candidates = reranked_candidates[:TOP_K_FINAL]

#     # Phase 3: Expand each top candidate with previous and next pages,
#     # then rerank these expanded candidate-documents to pick the best expanded chunk.
#     expanded_candidates = [expand_with_neighbor_pages(cand, docs, neighbor_pages=1) for cand in top_candidates]

#     if len(expanded_candidates) > 1:
#         final_reranked = cross_encoder_reranking(query, expanded_candidates)
#         return final_reranked[:TOP_K_FINAL]
#     else:
#         return expanded_candidates[:TOP_K_FINAL]

# # -------------------------
# # LLM Answer Generation
# # -------------------------

# async def answer_with_context_optimized(query: str, candidates: List[Document]) -> str:
#     """Optimized answer generation using the best expanded candidate only."""
#     if not candidates:
#         return "I could not find relevant content in the document."

#     # Best candidate is the first (already reranked by hybrid_retrieval_optimized)
#     best_doc = candidates[0]

#     # Build assembled context from the expanded best_doc (it already includes neighbors)
#     assembled_context = best_doc.page_content

#     prompt = f"""Based EXCLUSIVELY on the following context, provide a concise and accurate answer to the question.

# Question: {query}

# Context Information:
# {assembled_context}

# Instructions:
# - Answer using ONLY information from the provided context
# - If the context doesn't contain relevant information, state "I cannot find this information in the document"
# - Be precise and cite the source pages when possible (use the [Page X] markers present in the context)
# - Keep the answer focused and avoid speculation

# Answer:"""

#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         return "❌ OpenAI API key not configured."

#     try:
#         client = openai.AsyncOpenAI(api_key=openai_api_key)
#         response = await client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a precise assistant that answers questions based strictly on provided context. Never hallucinate or use external knowledge."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.1,
#             max_tokens=800,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         logger.error(f"OpenAI API error: {e}")
#         return f"Error generating response: {e}"

# # -------------------------
# # Streamlit UI
# # -------------------------

# st.title("🚀 Ultra-Fast PDF Chat — Neighbor-aware Reranking (Ordered)")

# col1, col2 = st.columns([1, 2])

# with col1:
#     uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)

#     if uploaded_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tfile:
#             tfile.write(uploaded_file.read())
#             pdf_path = tfile.name

#         st.success("✅ PDF uploaded successfully")

#         if st.button("Process PDF", type="primary"):
#             with st.spinner("🚀 Ultra-fast PDF Processing..."):
#                 start_total = time.time()

#                 # 1. Fast PDF parsing
#                 parse_start = time.time()
#                 pages_markdown = asyncio.run(parse_pdf_ultra_fast(pdf_path))
#                 parse_time = time.time() - parse_start
#                 st.info(f"📄 Parsed {len(pages_markdown)} pages in {parse_time:.2f}s")

#                 # 2. Fast chunking with deduplication
#                 chunk_start = time.time()
#                 docs = chunk_documents_ultra_fast(pages_markdown, source=uploaded_file.name)
#                 chunk_time = time.time() - chunk_start
#                 st.info(f"✂️ Created {len(docs)} unique chunks in {chunk_time:.2f}s")

#                 # 3. Parallel indexing
#                 index_start = time.time()
#                 with ThreadPoolExecutor(max_workers=2) as executor:
#                     bm25_future = executor.submit(build_bm25_fast, docs)
#                     vector_future = executor.submit(build_vector_store_fast, docs)

#                     bm25_obj, tokenized_texts = bm25_future.result()
#                     vectorstore = vector_future.result()

#                 index_time = time.time() - index_start
#                 st.info(f"🔍 Indexing (embedding + BM25 build) initiated/completed in {index_time:.2f}s")

#                 total_time = time.time() - start_total

#                 st.session_state.update({
#                     "docs": docs,
#                     "vectorstore": vectorstore,
#                     "bm25_obj": bm25_obj,
#                     "tokenized_texts": tokenized_texts,
#                     "processed": True
#                 })

#                 st.success(f"✅ Total processing time: {total_time:.2f}s")

#                 with st.expander("Performance Details"):
#                     st.metric("PDF Parsing", f"{parse_time:.2f}s")
#                     st.metric("Chunking", f"{chunk_time:.2f}s")
#                     st.metric("Indexing (start/complete)", f"{index_time:.2f}s")
#                     st.metric("Total", f"{total_time:.2f}s")

#         if not st.session_state.get("processed") and os.path.isdir(INDEX_DIR):
#             if st.button("Load Existing Index"):
#                 try:
#                     vectorstore = FAISS.load_local(INDEX_DIR, load_embedding_model())
#                     st.session_state["vectorstore"] = vectorstore
#                     st.success("Loaded existing FAISS index")
#                 except Exception as e:
#                     st.error(f"Failed to load vectorstore: {e}")

# with col2:
#     if st.session_state.get("processed"):
#         query = st.text_input("Your question:", placeholder="Ask about the PDF content...")

#         if st.button("Get Answer") and query.strip():
#             with st.spinner("🔍 Advanced retrieval with reranking..."):
#                 start = time.time()
#                 candidates = hybrid_retrieval_optimized(
#                     query,
#                     st.session_state["docs"],
#                     st.session_state["vectorstore"],
#                     st.session_state["bm25_obj"],
#                     st.session_state["tokenized_texts"]
#                 )
#                 retrieval_time = time.time() - start

#                 col_context, col_meta, col_answer = st.columns([2, 1, 2])

#                 with col_context:
#                     st.subheader("📄 Retrieved Context (Expanded & Reranked)")
#                     for i, doc in enumerate(candidates):
#                         # show which pages were included
#                         expanded_pages = doc.metadata.get("expanded_pages", [doc.metadata.get("page")])
#                         header = f"Context {i+1} (Orig Page {doc.metadata.get('orig_page', doc.metadata.get('page', 'N/A'))}) — Expanded Pages: {expanded_pages}"
#                         with st.expander(header, expanded=(i == 0)):
#                             st.write(doc.page_content)
#                             st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")

#                 with col_meta:
#                     st.subheader("🔍 Retrieval Info")
#                     st.metric("Retrieval Time", f"{retrieval_time:.2f}s")
#                     st.metric("Returned Chunks", len(candidates))
#                     st.metric("Strategy", "Hybrid -> Rerank -> Expand (ordered) -> Rerank")

#                     for i, doc in enumerate(candidates):
#                         st.write(f"**Candidate {i+1}:**")
#                         st.write(f"Orig Page: {doc.metadata.get('orig_page', 'N/A')}")
#                         st.write(f"Expanded Pages: {doc.metadata.get('expanded_pages', [])}")
#                         st.write(f"Length: {doc.metadata.get('combined_chunk_length', len(doc.page_content))} chars (approx)")

#                 with col_answer:
#                     st.subheader("🤖 AI Response")
#                     answer_start = time.time()
#                     answer = asyncio.run(answer_with_context_optimized(query, candidates))
#                     answer_time = time.time() - answer_start

#                     st.write(answer)
#                     st.caption(f"Retrieval: {retrieval_time:.2f}s | Generation: {answer_time:.2f}s")

# # -------------------------
# # Sidebar management
# # -------------------------
# with st.sidebar:
#     st.header("⚙️ Management")

#     if st.button("Clear Cache & Session"):
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
#         if os.path.isdir(INDEX_DIR):
#             import shutil
#             shutil.rmtree(INDEX_DIR)
#         try:
#             st.cache_data.clear()
#             st.cache_resource.clear()
#         except Exception:
#             pass
#         st.success("Cleared cache & session state")
#         st.experimental_rerun()

#     if st.session_state.get("docs"):
#         st.download_button(
#             "📥 Export Metadata",
#             json.dumps([
#                 {
#                     "source": d.metadata.get('source'),
#                     "page": d.metadata.get('page'),
#                     "chunk": d.metadata.get('chunk'),
#                     "text_preview": d.page_content[:200],
#                     "length": len(d.page_content)
#                 } for d in st.session_state["docs"][:100]
#             ], indent=2),
#             file_name="document_metadata.json",
#             mime="application/json"
#         )

#     if st.sidebar.checkbox("Show Performance Info"):
#         if st.session_state.get("docs"):
#             st.sidebar.metric("Total Chunks", len(st.session_state["docs"]))
#             avg_len = np.mean([len(d.page_content) for d in st.session_state["docs"]])
#             st.sidebar.metric("Avg Chunk Length", f"{avg_len:.0f} chars")

# # Cleanup temporary file
# if 'pdf_path' in locals() and os.path.exists(pdf_path):
#     try:
#         os.unlink(pdf_path)
#     except Exception:
#         pass


"""
Ultra-Fast PDF Chat: PyMuPDF + LangChain + FAISS
Updated: Powerful reranking for expanded chunks with neighbor context
"""

import os
import tempfile
import json
from typing import List, Tuple, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
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
import threading
import re

# -------------------------
# Optimized Configuration
# -------------------------
load_dotenv()
st.set_page_config(page_title="🚀 Ultra-Fast PDF Chat", layout="wide")

# Model and processing settings
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight reranker
CHUNK_SIZE = 2000  # Reduced for better precision
CHUNK_OVERLAP = 200
TOP_K = 10  # Retrieve more initially for reranking
TOP_K_FINAL = 3  # Final number after reranking

INDEX_DIR = "faiss_index_dir"

# OCR optimization
TESSERACT_CONFIG = r'--oem 3 --psm 3 -c preserve_interword_spaces=1 tessedit_do_invert=0'

# Parallel processing settings
MAX_WORKERS = min(8, os.cpu_count() or 4)
EMBEDDING_BATCH_SIZE = 32  # Reduced to avoid timeout and rate limits

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
    1. pymupdf4llm markdown attempt (fallback if rich text)
    2. Native PyMuPDF text (fast)
    3. Fast OCR fallback (only if necessary)
    """
    # 1) Try pymupdf4llm markdown for richer formatting (kept as fallback)
    try:
        temp_doc = fitz.open()
        try:
            temp_doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
            markdown_text = pymupdf4llm.to_markdown(temp_doc)
            if markdown_text and len(markdown_text.strip()) > 50:
                return markdown_text, True
        finally:
            temp_doc.close()
    except Exception as e:
        logger.debug(f"pymupdf4llm failed for page {page.number}: {e}")

    # 2) Try native text extraction (fast, low overhead)
    try:
        native_text = page.get_text("text", sort=True)
        if native_text:
            native_text_clean = native_text.replace("-\n", "")
            if len(native_text_clean.strip()) > 50:
                return native_text_clean, True
    except Exception as e:
        logger.debug(f"Native text extraction failed for page {page.number}: {e}")

    # 3) Quick OCR check - only if page has images and no text found
    try:
        image_list = page.get_images()
        if image_list:
            pix = page.get_pixmap(matrix=fitz.Matrix(100/72, 100/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            arr = np.array(img)
            if arr.ndim == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_text = pytesseract.image_to_string(binary, config=TESSERACT_CONFIG)
            if ocr_text and len(ocr_text.strip()) > 30:
                return ocr_text, False
    except Exception as e:
        logger.debug(f"OCR failed for page {page.number}: {e}")

    return "", False

def process_page_batch(args: Tuple[str, List[int]]) -> List[Tuple[int, str]]:
    """Process multiple pages in batch for reduced overhead (thread worker)."""
    file_path, page_indices = args
    results = []
    try:
        with fitz.open(file_path) as doc:
            for page_num in page_indices:
                try:
                    page = doc[page_num]
                    text, _ = extract_text_optimized(page)
                    if text:
                        results.append((page_num, text))
                except Exception as e:
                    logger.debug(f"Page {page_num} failed: {e}")
                    continue
    except Exception as e:
        logger.debug(f"Failed opening doc in worker for {file_path}: {e}")
    return results

async def parse_pdf_ultra_fast(file_path: str) -> List[Tuple[int, str]]:
    """Ultra-fast PDF parsing with optimized thread parallelization."""
    start_time = time.time()
    with fitz.open(file_path) as doc:
        total_pages = doc.page_count

    page_indices_all = list(range(total_pages))
    batch_count = max(1, min(MAX_WORKERS, total_pages))
    per_batch = (total_pages + batch_count - 1) // batch_count
    page_batches = []
    for i in range(0, total_pages, per_batch):
        page_batches.append((file_path, list(range(i, min(i + per_batch, total_pages)))))

    loop = asyncio.get_running_loop()
    results = []
    with ThreadPoolExecutor(max_workers=batch_count) as executor:
        tasks = [loop.run_in_executor(executor, process_page_batch, batch) for batch in page_batches]
        batch_results = await asyncio.gather(*tasks)
        for batch in batch_results:
            if batch:
                results.extend(batch)

    results.sort(key=lambda x: x[0])
    parsing_time = time.time() - start_time
    logger.info(f"PDF parsed {total_pages} pages in {parsing_time:.2f}s")
    return results

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

    docs: List[Document] = []
    for page_num, page_md_text in pages_markdown:
        clean_text = " ".join(page_md_text.split())
        if len(clean_text.strip()) < 40:
            continue

        chunks = splitter.split_text(clean_text)
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            metadata = {
                "source": source,
                "page": page_num + 1,  # 1-indexed pages
                "chunk": i,
                "chunk_length": len(chunk),
                "text_hash": hash(chunk)
            }
            docs.append(Document(page_content=chunk, metadata=metadata))

    # deduplicate
    unique_docs = []
    seen_hashes = set()
    for doc in docs:
        h = doc.metadata["text_hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    return unique_docs

def _save_faiss_in_thread(vectorstore: FAISS, index_path: str):
    """Save FAISS in separate thread to keep UI responsive."""
    try:
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)
        logger.info("FAISS saved to disk")
    except Exception as e:
        logger.warning(f"Failed to save FAISS index: {e}")

def build_vector_store_with_batching(_docs: List[Document], index_path: str = INDEX_DIR) -> FAISS:
    """FAISS index building with proper batching to avoid token limits."""
    embeddings = load_embedding_model()

    try:
        # Try direct method first
        vectorstore = FAISS.from_documents(_docs, embeddings)
    except Exception as e:
        if "max_tokens_per_request" in str(e) or "400" in str(e) or "rate_limit" in str(e).lower():
            st.warning("Document too large or rate limit hit, using manual batching...")
            vectorstore = _build_faiss_manual_batching(_docs, embeddings, index_path)
        else:
            raise e

    saver = threading.Thread(target=_save_faiss_in_thread, args=(vectorstore, index_path), daemon=True)
    saver.start()

    return vectorstore

def _build_faiss_manual_batching(docs: List[Document], embeddings, index_path: str) -> FAISS:
    """Manual batching for very large documents with better error handling."""
    from langchain_community.vectorstores.utils import DistanceStrategy

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    all_embeddings = []
    successful_texts = []
    successful_metadatas = []
    batch_size = EMBEDDING_BATCH_SIZE

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            status_text.text(f"Embedding batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
            progress_bar.progress(min((i + batch_size) / total, 1.0))
            
            try:
                batch_embeddings = embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                successful_texts.extend(batch_texts)
                successful_metadatas.extend(batch_metadatas)
                
            except Exception as e:
                logger.warning(f"Batch {i} failed: {e}, trying smaller batches...")
                # Try individual documents in the failed batch
                for j, text in enumerate(batch_texts):
                    try:
                        single_embedding = embeddings.embed_documents([text])
                        all_embeddings.extend(single_embedding)
                        successful_texts.append(text)
                        successful_metadatas.append(batch_metadatas[j])
                    except Exception as e2:
                        logger.error(f"Failed to embed single document: {e2}")
                        continue

        if not successful_texts:
            raise Exception("No documents could be embedded successfully")

        # Build FAISS index with successful embeddings only
        embedding_matrix = np.array(all_embeddings).astype('float32')
        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        # Create documents from successful embeddings
        successful_docs = []
        for text, metadata in zip(successful_texts, successful_metadatas):
            successful_docs.append(Document(page_content=text, metadata=metadata))

        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=FAISS._build_docstore(successful_docs),
            index_to_docstore_id=FAISS._build_index_to_docstore_id(successful_docs),
            distance_strategy=DistanceStrategy.COSINE
        )
        
        st.info(f"Successfully embedded {len(successful_docs)} out of {len(docs)} documents")
        
    except Exception as e:
        logger.error(f"FAISS manual batching failed: {e}")
        raise e
    finally:
        progress_bar.empty()
        status_text.empty()

    saver = threading.Thread(target=_save_faiss_in_thread, args=(vectorstore, index_path), daemon=True)
    saver.start()

    return vectorstore

@st.cache_resource
def build_vector_store_fast(_docs: List[Document], index_path: str = INDEX_DIR) -> FAISS:
    """Fast FAISS index building with batching (cached)."""
    return build_vector_store_with_batching(_docs, index_path)

def build_bm25_fast(docs: List[Document]) -> Tuple[BM25Okapi, List[str]]:
    """Optimized BM25 corpus building (faster tokenization)."""
    tokenized_corpus = []
    clean_texts = []
    for doc in docs:
        txt = doc.page_content.strip().lower()
        tokens = txt.split()
        if len(tokens) > 3:
            tokenized_corpus.append(tokens)
            clean_texts.append(" ".join(tokens))
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, clean_texts

# -------------------------
# Helper: Expand candidate with prev/next pages (preserve order prev -> candidate -> next)
# -------------------------
def expand_with_neighbor_pages(candidate: Document, all_docs: List[Document], neighbor_pages: int = 1) -> Document:
    """
    Given a candidate doc (one chunk), create a new Document which contains:
    - chunks from previous pages (ascending order)
    - the candidate chunk (in the middle)
    - chunks from next pages (ascending order)
    Example: if candidate page is 2 and neighbor_pages=1 -> order: page1, page2, page3
    """
    try:
        src = candidate.metadata.get("source")
        page = candidate.metadata.get("page", None)
        if page is None:
            return candidate  # nothing to expand

        # compute pages to include
        start_page = max(1, page - neighbor_pages)
        end_page = page + neighbor_pages
        pages_to_include = list(range(start_page, end_page + 1))

        # collect chunks by page (preserve chunk order within a page)
        page_to_chunks: Dict[int, List[Tuple[int, str]]] = {}
        for doc in all_docs:
            if doc.metadata.get("source") != src:
                continue
            doc_page = doc.metadata.get("page", None)
            if doc_page in pages_to_include:
                chunk_idx = doc.metadata.get("chunk", 0)
                page_to_chunks.setdefault(doc_page, []).append((chunk_idx, doc.page_content))

        # sort chunks within each page by chunk index
        for p in list(page_to_chunks.keys()):
            page_to_chunks[p].sort(key=lambda x: x[0])

        # Build ordered combined_text: previous pages in ascending order, candidate page chunks (with candidate chunk placed in its natural place), then next pages
        ordered_blocks: List[str] = []
        included_pages = set()

        for p in pages_to_include:
            chunks = page_to_chunks.get(p, [])
            if not chunks:
                continue
            # for each chunk in page, append with header
            for idx, content in chunks:
                header = f"[Page {p} Chunk {idx}]"
                ordered_blocks.append(f"{header}\n{content}")
            included_pages.add(p)

        # Ensure candidate chunk marker is present and in correct relative position:
        # The above loop already adds candidate page chunks in ascending order; nothing else needed.
        combined = "\n\n".join(ordered_blocks)

        new_meta = dict(candidate.metadata)
        new_meta["expanded_pages"] = sorted(list(included_pages))
        new_meta["expanded"] = True
        new_meta["orig_page"] = page
        new_meta["combined_chunk_length"] = len(combined)

        return Document(page_content=combined, metadata=new_meta)
    except Exception as e:
        logger.debug(f"Failed to expand candidate: {e}")
        return candidate

# -------------------------
# POWERFUL RERANKING SYSTEM FOR EXPANDED CHUNKS
# -------------------------

def extract_relevant_sections(expanded_content: str, query: str, orig_page: int = None) -> str:
    """
    Extract the most relevant sections from expanded content for better reranking.
    This helps the reranker focus on the most important parts.
    It now also boosts sections from the original page of the candidate.
    """
    try:
        # Split by page markers
        page_sections = re.split(r'(\[Page \d+ Chunk \d+\])', expanded_content)
        # The split regex now captures the delimiter, so we need to re-pair them
        processed_sections = []
        # The first element might be empty if the content starts with a marker
        start_idx = 1 if not page_sections[0].strip() and len(page_sections) > 1 else 0
        for i in range(start_idx, len(page_sections), 2):
            marker = page_sections[i]
            content = page_sections[i+1] if i+1 < len(page_sections) else ""
            processed_sections.append((marker, content))

        # Score each section based on query relevance
        query_terms = set(query.lower().split())
        scored_sections = []
        
        for marker, section in processed_sections:
            if not section.strip():
                continue
                
            section_lower = section.lower()
            # Calculate relevance score
            term_matches = sum(1 for term in query_terms if term in section_lower)
            
            # Extract page number from marker
            page_match = re.search(r'\[Page (\d+)', marker)
            section_page = int(page_match.group(1)) if page_match else None
            
            # Proximity bonus (can be more sophisticated)
            proximity_bonus = 1.0
            
            # Boost for sections from the original page
            orig_page_boost = 1.5 if orig_page and section_page == orig_page else 1.0
            
            # Bonus for sections that contain multiple query terms
            if len(query_terms) > 0:
                coverage = term_matches / len(query_terms)
            else:
                coverage = 0
                
            # Bonus for exact phrase matches
            exact_phrase_bonus = 2.0 if query.lower() in section_lower else 1.0
            
            score = (term_matches * 2 + coverage * 3 + exact_phrase_bonus) * proximity_bonus * orig_page_boost
            scored_sections.append((score, marker, section))
        
        # Sort by score and take top sections
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        top_sections = scored_sections[:3]  # Take top 3 most relevant sections
        
        # Reconstruct focused content
        focused_content = ""
        for score, marker, section in top_sections:
            focused_content += f"{marker}\n{section}\n\n"
            
        return focused_content.strip() if focused_content else expanded_content
        
    except Exception as e:
        logger.debug(f"Section extraction failed: {e}")
        return expanded_content

def enhanced_cross_encoder_reranking(query: str, candidates: List[Document]) -> List[Document]:
    """
    Enhanced reranking that considers both the full expanded context
    and focused relevant sections for better accuracy.
    """
    try:
        if not candidates:
            return []

        reranker = load_reranker_model()
        
        # Create two sets of pairs for more comprehensive reranking
        full_pairs = [(query, doc.page_content) for doc in candidates]
        
        # Create focused pairs using extracted relevant sections, passing original page for context
        focused_contents = [extract_relevant_sections(doc.page_content, query, doc.metadata.get('orig_page')) for doc in candidates]
        focused_pairs = [(query, focused_content) for focused_content in focused_contents]
        
        # Get scores for both full and focused content
        full_scores = []
        focused_scores = []
        
        batch_size = 16
        
        # Score full content
        for i in range(0, len(full_pairs), batch_size):
            batch_pairs = full_pairs[i:i + batch_size]
            batch_scores = reranker.predict(batch_pairs)
            full_scores.extend(batch_scores)
        
        # Score focused content
        for i in range(0, len(focused_pairs), batch_size):
            batch_pairs = focused_pairs[i:i + batch_size]
            batch_scores = reranker.predict(batch_pairs)
            focused_scores.extend(batch_scores)
        
        # Combine scores (weight focused content higher since it's more relevant)
        combined_scores = []
        for full_score, focused_score in zip(full_scores, focused_scores):
            # Weight focused content more heavily as it's directly relevant to query
            combined_score = (full_score * 0.4) + (focused_score * 0.6)
            combined_scores.append(combined_score)
        
        scored_docs = list(zip(candidates, combined_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Reranking completed: {[(score, doc.metadata.get('orig_page', 'N/A')) for doc, score in scored_docs[:3]]}")
        
        return [doc for doc, _ in scored_docs]
        
    except Exception as e:
        logger.warning(f"Enhanced reranking failed: {e}, falling back to basic reranking")
        # Fallback to basic reranking
        return cross_encoder_reranking(query, candidates)

def cross_encoder_reranking(query: str, candidates: List[Document]) -> List[Document]:
    """Use cross-encoder for precise reranking of top candidates. Batches predictions for speed."""
    try:
        if not candidates:
            return []

        reranker = load_reranker_model()
        pairs = [(query, doc.page_content) for doc in candidates]

        batch_size = 16
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = reranker.predict(batch_pairs)
            scores.extend(batch_scores)

        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]
    except Exception as e:
        logger.warning(f"Reranking failed: {e}, falling back to original ranking")
        return candidates

def hybrid_retrieval_optimized(query: str, docs: List[Document],
                              vectorstore: FAISS, bm25_obj: BM25Okapi,
                              tokenized_texts: List[str], top_k: int = TOP_K) -> List[Document]:
    """
    Optimized hybrid retrieval with POWERFUL reranking for expanded chunks.
    Now uses enhanced reranking that considers both full context and focused sections.
    """
    # Phase 1: Initial retrieval (more candidates)
    initial_k = min(top_k * 3, len(docs))

    # Dense retrieval
    dense_results = []
    dense_scores = []
    try:
        dense_docs_with_scores = vectorstore.similarity_search_with_score(query, k=initial_k)
        if dense_docs_with_scores:
            dense_results, dense_scores = zip(*dense_docs_with_scores)
        else:
            dense_results, dense_scores = [], []
    except Exception as e:
        logger.warning(f"Dense retrieval failed: {e}")
        dense_results, dense_scores = [], []

    # Sparse retrieval using BM25
    query_tokens = query.lower().split()
    if len(query_tokens) == 0:
        bm25_scores = np.zeros(len(tokenized_texts))
    else:
        bm25_scores = bm25_obj.get_scores(query_tokens)

    # Normalize BM25 scores safely
    if len(bm25_scores) > 0:
        max_bm25 = np.max(bm25_scores)
        bm25_scores = bm25_scores / max_bm25 if max_bm25 > 0 else np.zeros_like(bm25_scores)

    # Combine initial scores (use small dictionary keyed by doc tuple)
    candidate_scores = {}

    # add dense results (weight semantic higher)
    for doc, score in zip(dense_results, dense_scores):
        key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk'))
        try:
            dense_norm_score = float(score)
            dense_norm_score = 1 / (1 + np.exp(-dense_norm_score))
        except Exception:
            dense_norm_score = 0.5
        candidate_scores[key] = candidate_scores.get(key, 0) + 0.65 * dense_norm_score

    # add sparse results (keyword matching)
    if len(bm25_scores) > 0:
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:initial_k]
        for idx in top_bm25_indices:
            if idx < len(docs):
                doc = docs[idx]
                key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk'))
                candidate_scores[key] = candidate_scores.get(key, 0) + 0.35 * float(bm25_scores[idx])

    # convert candidate_scores back to document list (keep top initial_k)
    scored_candidates = []
    for key, score in candidate_scores.items():
        for doc in docs:
            if (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('chunk')) == key:
                scored_candidates.append((doc, score))
                break

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    initial_candidates = [doc for doc, _ in scored_candidates[:initial_k]]

    # Phase 2: Cross-encoder reranking for precision (first pass)
    max_rerank = max(10, TOP_K_FINAL * 4)
    rerank_slice = initial_candidates[:max_rerank]

    if len(rerank_slice) > 1:
        reranked_candidates = cross_encoder_reranking(query, rerank_slice)
    else:
        reranked_candidates = rerank_slice

    # Keep top-K_FINAL from first reranking
    top_candidates = reranked_candidates[:TOP_K_FINAL]

    # Phase 3: Expand each top candidate with previous and next pages
    expanded_candidates = [expand_with_neighbor_pages(cand, docs, neighbor_pages=1) for cand in top_candidates]

    # POWERFUL FINAL RERANKING: Use enhanced reranking on expanded chunks
    if len(expanded_candidates) > 1:
        # Use ENHANCED reranking that considers both full context and focused sections
        final_reranked = enhanced_cross_encoder_reranking(query, expanded_candidates)
        return final_reranked[:TOP_K_FINAL]
    else:
        return expanded_candidates[:TOP_K_FINAL]

# -------------------------
# LLM Answer Generation - IMPROVED VERSION
# -------------------------

async def answer_with_context_optimized(query: str, candidates: List[Document]) -> str:
    """Optimized answer generation using the best expanded candidate only."""
    if not candidates:
        return "I could not find relevant content in the document."

    # Use the first candidate (which is now properly reranked after expansion)
    best_doc = candidates[0]

    # Build assembled context from the expanded best_doc (it already includes neighbors)
    assembled_context = best_doc.page_content

    prompt = f"""Based EXCLUSIVELY on the following context, provide a concise and accurate answer to the question.

Question: {query}

Context Information:
{assembled_context}

Instructions:
- Answer using ONLY information from the provided context
- If the context doesn't contain relevant information, state "I cannot find this information in the document"
- Be precise and cite the source pages when possible (use the [Page X] markers present in the context)
- Keep the answer focused and avoid speculation
- If multiple pages are relevant, synthesize information from all of them

Answer:"""

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "❌ OpenAI API key not configured."

    try:
        client = openai.AsyncOpenAI(api_key=openai_api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise assistant that answers questions based strictly on provided context. Never hallucinate or use external knowledge. Synthesize information from multiple pages when relevant."},
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

st.title("🚀 Ultra-Fast PDF Chat — POWERFUL Reranking for Expanded Chunks")

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

                # 3. Parallel indexing with better error handling
                index_start = time.time()
                try:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        bm25_future = executor.submit(build_bm25_fast, docs)
                        vector_future = executor.submit(build_vector_store_fast, docs)

                        bm25_obj, tokenized_texts = bm25_future.result()
                        vectorstore = vector_future.result()

                    index_time = time.time() - index_start
                    st.info(f"🔍 Indexing (embedding + BM25 build) completed in {index_time:.2f}s")

                    total_time = time.time() - start_total

                    st.session_state.update({
                        "docs": docs,
                        "vectorstore": vectorstore,
                        "bm25_obj": bm25_obj,
                        "tokenized_texts": tokenized_texts,
                        "processed": True
                    })

                    st.success(f"✅ Total processing time: {total_time:.2f}s")

                    with st.expander("Performance Details"):
                        st.metric("PDF Parsing", f"{parse_time:.2f}s")
                        st.metric("Chunking", f"{chunk_time:.2f}s")
                        st.metric("Indexing", f"{index_time:.2f}s")
                        st.metric("Total", f"{total_time:.2f}s")
                        
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    logger.error(f"Indexing error: {e}")

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
            with st.spinner("🔍 POWERFUL retrieval with enhanced reranking..."):
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
                    st.subheader("📄 Retrieved Context (Enhanced Reranking)")
                    for i, doc in enumerate(candidates):
                        # show which pages were included
                        expanded_pages = doc.metadata.get("expanded_pages", [doc.metadata.get("page")])
                        header = f"Context {i+1} (Orig Page {doc.metadata.get('orig_page', doc.metadata.get('page', 'N/A'))}) — Expanded Pages: {expanded_pages}"
                        with st.expander(header, expanded=(i == 0)):
                            st.write(doc.page_content)
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")

                with col_meta:
                    st.subheader("🔍 Retrieval Info")
                    st.metric("Retrieval Time", f"{retrieval_time:.2f}s")
                    st.metric("Returned Chunks", len(candidates))
                    st.metric("Strategy", "Hybrid → Rerank → Expand → POWERFUL Rerank")

                    for i, doc in enumerate(candidates):
                        st.write(f"**Candidate {i+1}:**")
                        st.write(f"Orig Page: {doc.metadata.get('orig_page', 'N/A')}")
                        st.write(f"Expanded Pages: {doc.metadata.get('expanded_pages', [])}")
                        st.write(f"Length: {doc.metadata.get('combined_chunk_length', len(doc.page_content))} chars (approx)")

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
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        st.success("Cleared cache & session state")
        st.experimental_rerun()

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
    try:
        os.unlink(pdf_path)
    except Exception:
        pass