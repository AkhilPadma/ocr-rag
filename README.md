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
<img width="1472" height="1520" alt="image" src="https://github.com/user-attachments/assets/58506e4f-6ae9-446b-920a-d32d8420f68e" />

<img width="1472" height="1160" alt="image" src="https://github.com/user-attachments/assets/f3e4eae5-0203-4c33-b878-dbbee73cdde1" />




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
