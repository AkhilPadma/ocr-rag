# Multimodal Typography RAG

A multimodal Retrieval-Augmented Generation (RAG) system built over scanned historical typography catalogs. Upload a scanned PDF, ask questions in natural language, and get answers with page citations and retrieved specimen images.

Built as a portfolio project targeting the Monotype Solutions Research Engineer role — the system handles real OCR challenges on aged/sepia-toned scans and routes queries across text and visual retrieval paths.

---

## What it does

- Upload any scanned typography PDF
- Ask text questions: *"What point sizes is Cooper Black available in?"*
- Ask visual questions: *"Show me a bold typeface specimen"*
- Ask hybrid questions: *"What Century variants do they sell and show me one"*
- Get answers with page citations + retrieved page images

---

## Architecture

```
PDF Upload
    │
    ▼
OCR Pipeline (PyMuPDF + OpenCV preprocessing + pytesseract)
    │
    ├── Text path: chunks → BGE-small embeddings → Chroma
    │                        BM25 sparse index
    │
    └── Image path: page images → CLIP embeddings → Chroma
    │
    ▼
LLM Query Router (Groq qwen3.8-27b)
    │
    ├── TEXT  → Hybrid Retrieval (BGE + BM25 + RRF) → Cross-Encoder Reranking → LLM Answer
    ├── IMAGE → CLIP similarity search → Retrieved page images
    └── BOTH  → Both paths merged
    │
    ▼
Streamlit UI (answer + page citations + specimen images)
```

---

## Key engineering decisions and tradeoffs

**OCR preprocessing — why not just pytesseract directly?**
The corpus is aged sepia-toned scans. Raw Otsu binarization inverts the image (mean pixel < 128 = dark background). We detect inversion, correct it, then run both Otsu and adaptive thresholding, picking the method with more white pixels. Without this, tesseract returns zero characters.

**Hybrid retrieval — why BGE + BM25, not one or the other?**
Dense embeddings catch semantic queries ("script typefaces") but miss exact keyword matches ("Cooper Black No. 282" — garbled by OCR to "Cooper Black Na, 282"). BM25 catches exact tokens but misses semantic variation. RRF fuses both ranked lists: `score = 1/(k + dense_rank) + 1/(k + bm25_rank)`. Together they improve named font retrieval from rank 2 to rank 1.

**Cross-encoder reranking — why not just use the bi-encoder scores?**
Bi-encoders (BGE-small) encode query and document independently — fast but less precise. Cross-encoders read query + document jointly, catching subtle relevance signals. We retrieve top-20 with the bi-encoder (cheap) and rerank with a cross-encoder (accurate) to get top-5. Tradeoff: adds ~200-400ms per query.

**CLIP for image retrieval — why does it work?**
CLIP was trained on 400M image-caption pairs so text and image embeddings live in the same vector space. "Show me a bold slab-serif font" encodes as a text vector; we measure cosine similarity against every page image's CLIP vector. Limitation: CLIP was not trained on typography specifically — off-the-shelf similarity scores are 0.23-0.30. Fine-tuning on a labelled typography dataset would push this higher (Phase 2).

**Why 200 DPI not 300 or 600?**
200 DPI gives ~2-3s per page OCR with acceptable accuracy on body text and font labels. 300 DPI doubles processing time with marginal accuracy gains for our content. 600 DPI is overkill — no accuracy benefit justifies 10x the time. The size tables (tiny numbers in cramped grids) still OCR poorly at 200 DPI — noted as a known limitation.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, handles both native text and image rendering |
| OCR | pytesseract + OpenCV | CPU-based, no GPU needed for 20-page dev runs |
| Text embeddings | BAAI/bge-small-en-v1.5 | 384-dim, ~90MB, strong performance/size tradeoff |
| Image embeddings | clip-ViT-B-32 | Standard CLIP, 512-dim, runs on CPU |
| Sparse retrieval | BM25Okapi (rank-bm25) | Exact keyword matching, zero setup cost |
| Vector store | Chroma | Persistent, local, no server needed |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | ~22MB, strong reranking quality |
| LLM | Groq qwen/qwen3.8-27b | Free tier, fast inference, handles OCR noise |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respects natural text boundaries |
| UI | Streamlit | Fast to build, easy to demo |

---

## Setup

**Requirements:**
- Python 3.11
- Tesseract OCR binary (system-level install)
- Groq API key (free at console.groq.com)

**Install Tesseract (Windows):**
Download from https://github.com/UB-Mannheim/tesseract/wiki and install.

**Project setup:**
```bash
git clone https://github.com/AkhilPadma/ocr-rag
cd ocr-rag

python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

**Environment variables — create `.env`:**
```
GROQ_API_KEY=your_key_here
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
GROQ_MODEL=qwen/qwen3.8-27b
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

**Run:**
```bash
streamlit run app.py --server.maxUploadSize 1024
```

Upload a scanned typography PDF from the sidebar and start querying.

---

## Demo queries

| Query | Route | What happens |
|---|---|---|
| "What point sizes is Cooper Black available in?" | TEXT | Retrieves page 17, answers with 7 sizes |
| "Show me a bold typeface specimen" | IMAGE | Returns top-3 matching page images |
| "What Century typeface variants are available?" | TEXT | Lists all Century family entries |
| "Show me script fonts" | IMAGE | Returns pages with script specimen entries |

=

## Corpus

[Los Angeles Type Founders Specimen Book, early 1970s](https://archive.org/details/LATFSpecimenEarly1970s) — public domain, Internet Archive.
