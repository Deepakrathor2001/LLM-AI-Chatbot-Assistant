"""
Vision AI - RAG Engine
PDF ingestion, chunking, Gemini embeddings, ChromaDB retrieval.
Uses only google-genai (no LangChain).
"""

import os
import hashlib
from pathlib import Path
from typing import Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import chromadb
from chromadb.config import Settings
from google import genai
from pypdf import PdfReader

BASE_DIR        = Path(__file__).resolve().parent
CHROMA_DIR      = BASE_DIR / "data" / "chroma_db"
UPLOADS_DIR     = BASE_DIR / "data" / "uploads"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
COLLECTION_NAME = "visionai_documents"


def _client(api_key: str):
    return genai.Client(api_key=api_key)


def _embed_texts(texts: list, api_key: str) -> list:
    c        = _client(api_key)
    response = c.models.embed_content(model="models/embedding-001", contents=texts)
    items    = getattr(response, "embeddings", None) or []
    result   = []
    for item in items:
        v = getattr(item, "values", None) or getattr(item, "embedding", None) or []
        result.append(v)
    return result


def _embed_query(query: str, api_key: str) -> list:
    vecs = _embed_texts([query], api_key)
    return vecs[0] if vecs else []


def _load_pdf(path: Path) -> list:
    reader = PdfReader(str(path))
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def _chunk(text: str) -> list:
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + int(CHUNK_SIZE * 0.55):
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


def _chroma():
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()[:16]


def ingest_pdf(pdf_path: Path, api_key: str) -> dict:
    try:
        file_id = _hash(pdf_path)
        pages   = _load_pdf(pdf_path)
        chunks  = []
        for pg in pages:
            for txt in _chunk(pg["text"]):
                chunks.append({"page": pg["page"], "text": txt})
        if not chunks:
            return {"status": "error", "message": "No text extracted."}

        texts     = [c["text"] for c in chunks]
        metas     = [{"source": pdf_path.name, "file_id": file_id,
                      "page": str(c["page"]), "chunk_index": str(i)}
                     for i, c in enumerate(chunks)]
        ids       = [f"{file_id}_c{i}" for i in range(len(chunks))]
        embeddings = _embed_texts(texts, api_key)
        if not embeddings or len(embeddings) != len(texts):
            return {"status": "error", "message": "Embedding failed."}

        col = _collection(_chroma())
        col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
        return {"status": "success", "file": pdf_path.name,
                "chunks": len(chunks), "pages": len(pages), "file_id": file_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def retrieve_context(query: str, api_key: str, top_k: int = 5,
                     source_filter: Optional[str] = None) -> list:
    try:
        qv  = _embed_query(query, api_key)
        if not qv:
            return []
        col = _collection(_chroma())
        if col.count() == 0:
            return []
        where   = {"source": source_filter} if source_filter else None
        results = col.query(
            query_embeddings=[qv],
            n_results=min(top_k, col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(results["documents"][0],
                                    results["metadatas"][0],
                                    results["distances"][0]):
            score = round(1 - dist, 3)
            if score > 0.25:
                out.append({"text": doc, "source": meta.get("source", "?"),
                            "page": meta.get("page", "?"), "score": score})
        return sorted(out, key=lambda x: x["score"], reverse=True)
    except Exception:
        return []


def list_indexed_documents(api_key: str) -> list:
    try:
        col = _collection(_chroma())
        if col.count() == 0:
            return []
        res  = col.get(include=["metadatas"])
        seen = {}
        for meta in res["metadatas"]:
            fid = meta.get("file_id", "")
            if fid not in seen:
                seen[fid] = {"name": meta.get("source", "?"), "file_id": fid}
        return list(seen.values())
    except Exception:
        return []


def delete_document(file_id: str) -> bool:
    try:
        col = _collection(_chroma())
        res = col.get(where={"file_id": file_id}, include=["metadatas"])
        ids = res.get("ids", [])
        if ids:
            col.delete(ids=ids)
        return True
    except Exception:
        return False


def get_vector_store_stats() -> dict:
    try:
        col   = _collection(_chroma())
        count = col.count()
        docs  = list_indexed_documents(None)
        return {"total_chunks": count, "total_documents": len(docs)}
    except Exception:
        return {"total_chunks": 0, "total_documents": 0}
