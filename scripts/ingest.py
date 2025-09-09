\
from __future__ import annotations
import os, sys, argparse, time, hashlib, math
from typing import Iterable, List, Dict, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tqdm import tqdm

# Astra
from astrapy.db import AstraDB

"""
Ingestor: PDFs, URLs (HTML), and Markdown files → chunks → embeddings (Ollama) → Astra DB Vector collection.

Env required:
  ASTRA_DB_API_ENDPOINT
  ASTRA_DB_APP_TOKEN
  ASTRA_DB_KEYSPACE=default_keyspace
  ASTRA_COLLECTION=docs

Embedding (one of):
  OLLAMA_BASE_URL (default http://localhost:11434)
  OLLAMA_EMBED_MODEL (default all-minilm:latest)

Usage examples:
  python scripts/ingest.py --pdf_dir ./my_pdfs
  python scripts/ingest.py --urls https://example.com https://docs.python.org
  python scripts/ingest.py --md_dir ./notes
  python scripts/ingest.py --pdf_dir ./p --urls https://a.com --md_dir ./notes
"""

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts).strip()

def fetch_url_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "ingestor/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Remove script/style
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def read_markdown_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, max_tokens: int = 800, overlap: int = 100) -> List[str]:
    # naive tokenization by words (works okay for MiniLM)
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    while i < len(words):
        chunk = " ".join(words[i:i+max_tokens])
        chunks.append(chunk)
        i += step
    return chunks

def get_ollama_embedding(text: str, base_url: str, model: str) -> List[float]:
    # Ollama /api/embeddings
    url = f"{base_url.rstrip('/')}/api/embeddings"
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    emb = data.get("embedding")
    if not emb:
        raise RuntimeError(f"No embedding returned from Ollama. Response keys: {list(data.keys())}")
    return emb

def connect_astra() -> AstraDB:
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_APP_TOKEN")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
    assert endpoint and token, "Set ASTRA_DB_API_ENDPOINT and ASTRA_DB_APP_TOKEN env vars"
    return AstraDB(token=token, api_endpoint=endpoint, namespace=keyspace)

def ensure_collection(adb: AstraDB, collection: str, dim: int) -> None:
    adb.create_collection_if_not_exists(collection_name=collection, dimension=dim, metric="cosine", indexing="dense_vector")

def upsert_chunks(adb: AstraDB, collection: str, items: List[Dict]) -> None:
    # items: [{_id, $vector, text, source, meta:{...}}]
    col = adb.collection(collection)
    # Upsert in small batches
    B = 32
    for i in range(0, len(items), B):
        batch = items[i:i+B]
        col.insert_many(batch, options={"ordered": False})

def hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, help="Directory of PDFs")
    ap.add_argument("--md_dir", type=str, help="Directory of Markdown files")
    ap.add_argument("--urls", nargs="*", help="One or more URLs")
    ap.add_argument("--collection", type=str, default=os.getenv("ASTRA_COLLECTION", "docs"))
    ap.add_argument("--embed_model", type=str, default=os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:latest"))
    ap.add_argument("--ollama", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args()

    # Gather documents
    docs: List[Tuple[str, str]] = []  # (source_id, text)
    if args.pdf_dir and os.path.isdir(args.pdf_dir):
        for root, _, files in os.walk(args.pdf_dir):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    path = os.path.join(root, fn)
                    try:
                        text = read_pdf_text(path)
                        if text.strip():
                            docs.append((f"file://{os.path.abspath(path)}", text))
                    except Exception as e:
                        print(f"[WARN] PDF extract failed: {path}: {e}")

    if args.md_dir and os.path.isdir(args.md_dir):
        for root, _, files in os.walk(args.md_dir):
            for fn in files:
                if fn.lower().endswith((".md", ".markdown", ".txt")):
                    path = os.path.join(root, fn)
                    try:
                        text = read_markdown_text(path)
                        if text.strip():
                            docs.append((f"file://{os.path.abspath(path)}", text))
                    except Exception as e:
                        print(f"[WARN] Markdown read failed: {path}: {e}")

    if args.urls:
        for u in args.urls:
            try:
                text = fetch_url_text(u)
                if text.strip():
                    docs.append((u, text))
            except Exception as e:
                print(f"[WARN] URL fetch failed: {u}: {e}")

    if not docs:
        print("No documents found. Provide --pdf_dir, --md_dir, or --urls.")
        sys.exit(1)

    # Create chunks
    all_chunks: List[Tuple[str, str]] = []  # (source_id, chunk_text)
    for source, text in docs:
        chunks = chunk_text(text, max_tokens=args.max_tokens, overlap=args.overlap)
        for idx, ch in enumerate(chunks):
            all_chunks.append((f"{source}#chunk-{idx}", ch))

    print(f"Prepared {len(all_chunks)} chunks from {len(docs)} documents. Embedding via Ollama model '{args.embed_model}'.")

    # Connect Astra and ensure collection
    adb = connect_astra()
    # Probe embedding dim with a tiny example
    probe = get_ollama_embedding("hello world", base_url=args.ollama, model=args.embed_model)
    dim = len(probe)
    ensure_collection(adb, args.collection, dim)

    # Embed & upsert
    to_upsert = []
    for sid, chunk in tqdm(all_chunks, desc="Embedding & staging"):
        emb = get_ollama_embedding(chunk, base_url=args.ollama, model=args.embed_model)
        doc_id = hash_id(sid)
        to_upsert.append({
            "_id": doc_id,
            "$vector": emb,
            "text": chunk,
            "source": sid.split("#chunk-")[0],
            "meta": {"chunk_id": sid.split("#chunk-")[-1], "ingested_at": int(time.time())}
        })

    print(f"Upserting {len(to_upsert)} chunks to Astra collection '{args.collection}'…")
    upsert_chunks(adb, args.collection, to_upsert)
    print("Done.")
    
if __name__ == "__main__":
    main()
