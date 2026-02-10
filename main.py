import os
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from ebooklib import epub, ITEM_DOCUMENT
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DOC_PATH = os.getenv("DOC_PATH", "doc.pdf")  # путь к PDF/EPUB/TXT
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1500"))  # размер чанка при нарезке
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # перекрытие чанков
MAX_SNIPPET_CHARS = int(os.getenv("MAX_SNIPPET_CHARS", "600"))  # сколько максимум отдавать наружу
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

app = FastAPI(
    title="KB Retrieval API (PDF/EPUB snippets)",
    version="1.0.0",
    description="Search and return short relevant snippets from a document for GPT Actions."
)

# ----------------------------
# Models
# ----------------------------

class SearchHit(BaseModel):
    id: str
    score: float
    location: str
    snippet: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]

class ChunkResponse(BaseModel):
    id: str
    location: str
    text: str

# ----------------------------
# Text extraction
# ----------------------------

def extract_pdf_pages(path: str) -> List[Tuple[str, str]]:
    # returns list of (location, text)
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    pages: List[Tuple[str, str]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        loc = f"page:{i+1}"
        pages.append((loc, text))
    doc.close()
    return pages

def extract_epub_sections(path: str) -> List[Tuple[str, str]]:
    # optional; requires ebooklib + bs4
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(path)
    out: List[Tuple[str, str]] = []
    idx = 0
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            idx += 1
            soup = BeautifulSoup(item.get_content(), "lxml")
            text = soup.get_text("\n")
            loc = f"epub_section:{idx}"
            out.append((loc, text))
    return out

def extract_txt(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [("txt:1", f.read())]

def extract_document(path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document not found: {path}")

    lower = path.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_pages(path)
    if lower.endswith(".epub"):
        return extract_epub_sections(path)
    if lower.endswith(".txt"):
        return extract_txt(path)

    raise ValueError("Unsupported format. Use .pdf, .epub, or .txt")

# ----------------------------
# Chunking + simple retrieval
# ----------------------------

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s)]

def make_chunks(sections: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for loc, text in sections:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + CHUNK_CHARS)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "location": loc,
                    "text": chunk_text,
                    "tokens": tokenize(chunk_text)
                })
            if end == n:
                break
            start = max(0, end - CHUNK_OVERLAP)
    return chunks

def score_chunk(q_tokens: List[str], chunk_tokens: List[str]) -> float:
    # супер-простой скоринг: совпадения токенов + частота
    if not q_tokens or not chunk_tokens:
        return 0.0

    # частоты в чанке
    freq: Dict[str, int] = {}
    for t in chunk_tokens:
        freq[t] = freq.get(t, 0) + 1

    score = 0.0
    for qt in q_tokens:
        score += float(freq.get(qt, 0))
    # лёгкая нормализация по длине
    score /= (1.0 + (len(chunk_tokens) / 200.0))
    return score

def build_snippet(text: str, q: str, limit: int) -> str:
    # вырезаем небольшой фрагмент вокруг первого вхождения
    qn = q.strip().lower()
    if not qn:
        return (text[:limit] + "…") if len(text) > limit else text

    low = text.lower()
    idx = low.find(qn)
    if idx == -1:
        return (text[:limit] + "…") if len(text) > limit else text

    start = max(0, idx - 120)
    end = min(len(text), idx + 300)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"

    if len(snippet) > limit:
        snippet = snippet[:limit].rstrip() + "…"
    return snippet

# ----------------------------
# Load & index at startup
# ----------------------------

CHUNKS: List[Dict[str, Any]] = []

@app.on_event("startup")
def startup_load():
    global CHUNKS
    sections = extract_document(DOC_PATH)
    CHUNKS = make_chunks(sections)

@app.get("/health")
def health():
    return {"ok": True, "chunks": len(CHUNKS), "doc": DOC_PATH}

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), k: int = Query(TOP_K_DEFAULT, ge=1, le=10)):
    q_tokens = tokenize(q)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ch in CHUNKS:
        s = score_chunk(q_tokens, ch["tokens"])
        if s > 0:
            scored.append((s, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    results: List[SearchHit] = []
    for s, ch in top:
        results.append(SearchHit(
            id=ch["id"],
            score=round(float(s), 4),
            location=ch["location"],
            snippet=build_snippet(ch["text"], q, MAX_SNIPPET_CHARS)
        ))

    return SearchResponse(query=q, results=results)

@app.get("/chunk/{chunk_id}", response_model=ChunkResponse)
def get_chunk(chunk_id: str):
    for ch in CHUNKS:
        if ch["id"] == chunk_id:
            # отдаём ограниченный текст (чтобы не выдавать большие куски)
            text = ch["text"]
            if len(text) > MAX_SNIPPET_CHARS:
                text = text[:MAX_SNIPPET_CHARS].rstrip() + "…"
            return ChunkResponse(id=ch["id"], location=ch["location"], text=text)
    raise HTTPException(status_code=404, detail="Chunk not found")
