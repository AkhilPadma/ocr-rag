# config.py
# central place for all paths, model names, and settings
# if something needs changing (different model, different DPI) it changes here only

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- paths ---
BASE_DIR = Path(__file__).parent.parent  # D:\OCR Project\
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"

# --- tesseract ---
# pytesseract needs to know where the binary lives
TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# --- models ---
# text embedding model - 384 dim vectors, ~90MB, runs on CPU fine
TEXT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# CLIP model for image embedding - 512 dim vectors
# ViT-B/32 is the classic balance of speed vs quality
IMAGE_EMBED_MODEL = "clip-ViT-B-32"

# groq LLM for routing and generation
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- OCR settings ---
# 200 DPI is our sweet spot - good enough for body text and font labels
# 300 would be more accurate on tiny size tables but ~2x slower per page
# tradeoff: we accept slightly worse size-table OCR for 2x speed
OCR_DPI = 200

# minimum characters from native text extraction before we decide
# the page is image-only and fall back to OCR
# 50 chars = roughly one short line of text
NATIVE_TEXT_MIN_CHARS = 50

# how many pages to process in dev mode (None = all pages)
# set to 20 during development so OCR runs in under a minute
# change to None for final indexing run
DEV_PAGE_LIMIT = 20

# --- retrieval settings ---
# how many chunks to retrieve before reranking
RETRIEVAL_TOP_K = 20

# how many chunks to keep after reranking
RERANK_TOP_N = 5

# how many similar pages to retrieve for image queries
IMAGE_TOP_K = 3

# minimum chars for a page to be worth indexing
MIN_PAGE_CHARS = 50

# minimum quality score to index a page
MIN_QUALITY_SCORE = 0.1

 