# app.py
# Streamlit UI for the multimodal typography RAG system
# run with: streamlit run app.py

import streamlit as st
import sys
from pathlib import Path

# add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
import chromadb
import io
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

from src.config import (
    TESSERACT_PATH, OCR_DPI, NATIVE_TEXT_MIN_CHARS,
    TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL,
    GROQ_API_KEY, CHROMA_DIR,
    MIN_PAGE_CHARS, MIN_QUALITY_SCORE,
    RETRIEVAL_TOP_K, RERANK_TOP_N, IMAGE_TOP_K
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
GROQ_MODEL = "qwen/qwen3.8-27b"

# page config
st.set_page_config(
    page_title="Typography RAG",
    page_icon="T",
    layout="wide"
)

st.title("Multimodal Typography RAG")
st.caption("Query the Los Angeles Type Founders catalog using text and visual search")

# ----------------------------------------------------------------
# cached model loading - runs once, stays in memory across queries
# ----------------------------------------------------------------

@st.cache_resource
def load_models():
    text_embedder = SentenceTransformer(TEXT_EMBED_MODEL)
    clip_model = SentenceTransformer(IMAGE_EMBED_MODEL)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    groq_client = Groq(api_key=GROQ_API_KEY)
    return text_embedder, clip_model, reranker, groq_client

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    text_collection = client.get_collection("text_chunks")
    image_collection = client.get_collection("page_images")
    return text_collection, image_collection

# ----------------------------------------------------------------
# OCR pipeline - cached so PDF is only processed once per session
# ----------------------------------------------------------------

def preprocess_page_image(page, dpi=OCR_DPI):
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    if mean_val < 128:
        gray = cv2.bitwise_not(gray)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10
    )
    otsu_white = (binary_otsu == 255).mean()
    adaptive_white = (binary_adaptive == 255).mean()
    best = binary_otsu if otsu_white >= adaptive_white else binary_adaptive
    return pil_image, Image.fromarray(best)

def run_ocr(preprocessed_pil):
    ocr_text = pytesseract.image_to_string(preprocessed_pil, config='--psm 3')
    ocr_data = pytesseract.image_to_data(
        preprocessed_pil, config='--psm 3',
        output_type=pytesseract.Output.DICT
    )
    boxes = []
    for i, word in enumerate(ocr_data['text']):
        conf = int(ocr_data['conf'][i])
        if conf > 0 and word.strip():
            boxes.append({
                'word': word, 'x': ocr_data['left'][i],
                'y': ocr_data['top'][i], 'conf': conf
            })
    quality_score = sum(b['conf'] for b in boxes) / len(boxes) / 100 if boxes else 0.0
    return ocr_text, boxes, quality_score

@st.cache_data
def process_pdf(_pdf_bytes, pdf_name):
    # underscore prefix on _pdf_bytes tells streamlit not to hash the bytes
    doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
    pages_data = []
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=200, chunk_overlap=40, length_function=len,
    )
    all_chunks = []

    progress = st.progress(0, text="Processing PDF...")
    start_page = 1
    end_page = min(start_page + 20, len(doc))

    for idx, page_idx in enumerate(range(start_page, end_page)):
        page = doc[page_idx]
        page_num = page_idx + 1
        progress.progress((idx + 1) / (end_page - start_page),
                         text=f"OCR page {page_num}...")

        native_text = page.get_text().strip()
        if len(native_text) >= NATIVE_TEXT_MIN_CHARS:
            ocr_text, boxes, quality_score = native_text, [], 1.0
            text_source = "native"
        else:
            _, preprocessed = preprocess_page_image(page)
            ocr_text, boxes, quality_score = run_ocr(preprocessed)
            text_source = "ocr"

        zoom = OCR_DPI / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        page_image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        if len(ocr_text.strip()) < MIN_PAGE_CHARS and quality_score < MIN_QUALITY_SCORE:
            continue

        page_dict = {
            "page_num": page_num, "text_source": text_source,
            "ocr_text": ocr_text, "quality_score": quality_score,
            "page_image": page_image,
        }
        pages_data.append(page_dict)

        raw_chunks = splitter.split_text(ocr_text)
        for cidx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text) > 30:
                all_chunks.append({
                    "chunk_id": f"p{page_num}_c{cidx}",
                    "page_num": page_num, "text": chunk_text,
                    "quality_score": quality_score,
                    "char_count": len(chunk_text),
                })

    progress.empty()
    return pages_data, all_chunks

# ----------------------------------------------------------------
# retrieval functions
# ----------------------------------------------------------------

def hybrid_search(query, text_collection, text_embedder, bm25_index,
                  all_chunks, top_k=RETRIEVAL_TOP_K, rrf_k=60):
    query_embedding = text_embedder.encode(
        [query], normalize_embeddings=True).tolist()
    dense_results = text_collection.query(
        query_embeddings=query_embedding, n_results=top_k)
    dense_ranks = {cid: r+1 for r, cid in enumerate(dense_results['ids'][0])}

    query_tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(query_tokens)
    top_bm25 = bm25_scores.argsort()[::-1][:top_k]
    bm25_ranks = {all_chunks[i]["chunk_id"]: r+1
                  for r, i in enumerate(top_bm25)}

    all_ids = set(dense_ranks) | set(bm25_ranks)
    rrf_scores = {cid: (1/(rrf_k + dense_ranks.get(cid, 1000)) +
                        1/(rrf_k + bm25_ranks.get(cid, 1000)))
                  for cid in all_ids}

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    chunk_lookup = {c["chunk_id"]: c for c in all_chunks}
    results = []
    for cid, score in sorted_ids[:top_k]:
        if cid in chunk_lookup:
            r = chunk_lookup[cid].copy()
            r["rrf_score"] = score
            results.append(r)
    return results

def rerank_results(query, hybrid_results, reranker, top_n=RERANK_TOP_N):
    if not hybrid_results:
        return []
    pairs = [[query, r["text"]] for r in hybrid_results]
    scores = reranker.predict(pairs)
    for i, r in enumerate(hybrid_results):
        r["rerank_score"] = float(scores[i])
    return sorted(hybrid_results, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

def route_query(query, groq_client):
    prompt = f"""Classify this query into exactly one category:
- TEXT: asking for facts, names, prices, specifications
- IMAGE: asking to see or display something visually
- BOTH: needs facts AND visual display

Query: "{query}"
Reply with only one word: TEXT, IMAGE, or BOTH"""
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10, temperature=0,
    )
    route = resp.choices[0].message.content.strip().upper()
    return route if route in ["TEXT", "IMAGE", "BOTH"] else "BOTH"

def generate_answer(query, text_chunks, groq_client):
    if not text_chunks:
        return "No relevant text found."
    context = "\n\n".join(
        f"[Page {c['page_num']}]\n{c['text']}" for c in text_chunks)
    prompt = f"""You are an expert on the Los Angeles Type Founders catalog.
OCR noise: l2-pt=12pt, ld-pt=14pt, 24.pt=24pt, dots are artifacts.
Answer using ONLY the context below. Cite page numbers.
If not found say: Not found in indexed pages.

Context:
{context}

Question: {query}
Answer:"""
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.1,
    )
    return resp.choices[0].message.content.strip()

# ----------------------------------------------------------------
# main UI
# ----------------------------------------------------------------

# sidebar
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader(
        "Upload typography PDF", type=["pdf"],
        help="Upload the LATF specimen PDF"
    )
    st.divider()
    st.caption("Pipeline: OCR → Hybrid Retrieval (BGE + BM25 + RRF) → Cross-Encoder Reranking → LLM Generation")

# main area
if uploaded_file is None:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

# load models
with st.spinner("Loading models..."):
    text_embedder, clip_model, reranker, groq_client = load_models()
    text_collection, image_collection = load_chroma()

# process PDF
pdf_bytes = uploaded_file.read()
pages_data, all_chunks = process_pdf(pdf_bytes, uploaded_file.name)

# build BM25 index
corpus_tokens = [c["text"].lower().split() for c in all_chunks]
bm25_index = BM25Okapi(corpus_tokens)

st.success(f"Indexed {len(pages_data)} pages, {len(all_chunks)} chunks.")

# query input
st.divider()
query = st.text_input(
    "Ask a question",
    placeholder="e.g. What point sizes is Cooper Black available in?",
)

example_queries = [
    "What point sizes is Cooper Black available in?",
    "Show me a bold typeface specimen",
    "What Century typeface variants are available?",
    "Show me script fonts",
]

st.caption("Try:")
cols = st.columns(len(example_queries))
for i, eq in enumerate(example_queries):
    if cols[i].button(eq, use_container_width=True):
        query = eq

if not query:
    st.stop()

# run pipeline
with st.spinner("Searching..."):
    start = time.time()
    route = route_query(query, groq_client)

    text_results, image_results = [], []

    if route in ["TEXT", "BOTH"]:
        hybrid = hybrid_search(
            query, text_collection, text_embedder,
            bm25_index, all_chunks
        )
        text_results = rerank_results(query, hybrid, reranker)

    if route in ["IMAGE", "BOTH"]:
        qe = clip_model.encode([query], normalize_embeddings=True).tolist()
        img_res = image_collection.query(query_embeddings=qe, n_results=IMAGE_TOP_K)
        for meta, dist in zip(img_res['metadatas'][0], img_res['distances'][0]):
            pnum = meta["page_num"]
            match = next((p for p in pages_data if p["page_num"] == pnum), None)
            if match:
                image_results.append({
                    "page_num": pnum,
                    "similarity": 1 - dist,
                    "page_image": match["page_image"],
                })

    if text_results:
        answer = generate_answer(query, text_results, groq_client)
    else:
        answer = "Visual query — see images below."

    elapsed = time.time() - start

# display results
st.divider()
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Answer")
    # show route badge
    route_color = {"TEXT": "blue", "IMAGE": "green", "BOTH": "orange"}.get(route, "gray")
    st.markdown(f"Route: :{route_color}[{route}] | {elapsed:.1f}s")
    st.markdown(answer)

    if text_results:
        with st.expander(f"Sources ({len(text_results)} chunks)"):
            for r in text_results:
                st.caption(f"Page {r['page_num']} | rerank={r['rerank_score']:.2f}")
                st.text(r['text'][:200])
                st.divider()

with col2:
    if image_results:
        st.subheader("Retrieved Pages")
        for img_r in image_results:
            st.image(
                img_r["page_image"],
                caption=f"Page {img_r['page_num']} (similarity: {img_r['similarity']:.3f})",
                width=400,
            )