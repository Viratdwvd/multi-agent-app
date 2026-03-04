from __future__ import annotations
import os
import json
import time
import hashlib
from typing import List, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
import requests

from langflow_client import LangflowClient

# Optional Astra + PDF/HTML parsing for ingestion
from astrapy import DataAPIClient
from pypdf import PdfReader
from bs4 import BeautifulSoup

load_dotenv(override=True)

st.set_page_config(page_title="Multi‑Agent RAG (Langflow)", page_icon="🧩", layout="wide")
st.title("🧩 Multi‑Agent RAG • Langflow + Streamlit + Astra DB")

# ----------------------------
# Helpers for ingestion (minimal)
# ----------------------------
def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _chunk_text(text: str, max_tokens: int = 800, overlap: int = 100) -> List[str]:
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

def _read_pdf_bytes(file_bytes: bytes) -> str:
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts).strip()

def _read_markdown_bytes(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def _fetch_url_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "streamlit-uploader/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
    return "\n".join(lines)

def _get_ollama_embedding(text: str, base_url: str, model: str) -> List[float]:
    url = f"{base_url.rstrip('/')}/api/embeddings"
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    emb = data.get("embedding")
    if not emb:
        raise RuntimeError(f"No embedding returned from Ollama. Keys: {list(data.keys())}")
    return emb

def _connect_astra() -> AstraDB:
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_APP_TOKEN")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
    if not (endpoint and token):
        raise RuntimeError("Set ASTRA_DB_API_ENDPOINT and ASTRA_DB_APP_TOKEN in secrets or env.")
    return AstraDB(token=token, api_endpoint=endpoint, namespace=keyspace)

def _ensure_collection(adb: AstraDB, collection: str, dim: int) -> None:
    adb.create_collection_if_not_exists(collection_name=collection, dimension=dim, metric="cosine", indexing="dense_vector")

def _upsert_chunks(adb: AstraDB, collection: str, items: List[Dict]) -> None:
    col = adb.collection(collection)
    B = 32
    for i in range(0, len(items), B):
        batch = items[i:i+B]
        col.insert_many(batch, options={"ordered": False})

# ----------------------------
# Sidebar config & health
# ----------------------------
with st.sidebar:
    st.subheader("Langflow Config")
    base_url = st.text_input("Langflow Base URL", os.getenv("LANGFLOW_BASE_URL", ""))
    api_key = st.text_input("Langflow API Key", os.getenv("LANGFLOW_API_KEY", ""), type="password")
    flow_id = st.text_input("Flow ID", os.getenv("LANGFLOW_FLOW_ID", "multi-agent-rag"))

    client = LangflowClient(base_url=base_url, api_key=api_key, flow_id=flow_id)
    healthy = client.health()
    st.status("Langflow: Healthy" if healthy else "Langflow: Unreachable", state=("complete" if healthy else "error"))

    st.divider()
    st.subheader("Astra & Embeddings (for Upload → Ingest)")
    astra_endpoint = st.text_input("ASTRA_DB_API_ENDPOINT", os.getenv("ASTRA_DB_API_ENDPOINT", ""))
    astra_token = st.text_input("ASTRA_DB_APP_TOKEN", os.getenv("ASTRA_DB_APP_TOKEN", ""), type="password")
    astra_keyspace = st.text_input("ASTRA_DB_KEYSPACE", os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace"))
    astra_collection = st.text_input("ASTRA_COLLECTION", os.getenv("ASTRA_COLLECTION", "docs"))
    ollama_url = st.text_input("OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    embed_model = st.text_input("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:latest"))

    if st.button("Clear chat"):
        st.session_state.pop("history", None)
        st.rerun()

# Tabs: Chat | Upload & Ingest
tab_chat, tab_ingest = st.tabs(["💬 Chat", "📤 Upload & Ingest"])

# ----------------------------
# Chat tab
# ----------------------------
with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {role, content}

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask something…")
    if user_msg:
        st.session_state.history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    res = client.run(message=user_msg, history=st.session_state.history)
                    text = None
                    sources = []
                    try:
                        outputs = res.get("outputs", [])
                        if outputs and outputs[0].get("outputs"):
                            results = outputs[0]["outputs"][0].get("results", {})
                            text = (
                                results.get("message", {}).get("data", {}).get("content")
                                or results.get("text")
                            )
                            sources = results.get("sources") or results.get("documents") or []
                    except Exception:
                        pass

                    text = text or "(No text returned — ensure your flow ends with Chat/Text output)"
                    st.markdown(text)
                    if sources:
                        with st.expander("Sources"):
                            st.write(sources)
                except Exception as e:
                    st.error(f"Request failed: {e}")

        st.rerun()

# ----------------------------
# Upload & Ingest tab
# ----------------------------
with tab_ingest:
    st.markdown("Upload **PDF / Markdown / Text** files here and ingest them into your Astra DB collection used by the retriever in your Langflow flow.")
    files = st.file_uploader(
        "Choose files",
        type=["pdf", "md", "markdown", "txt"],
        accept_multiple_files=True
    )

    urls_str = st.text_input("Or paste URLs (comma-separated)", "")
    max_tokens = st.slider("Chunk size (approx words)", 300, 1200, 800, 50)
    overlap = st.slider("Chunk overlap (approx words)", 0, 300, 100, 10)

    disabled = not (astra_endpoint and astra_token and files or urls_str.strip())
    if st.button("Ingest to Astra", disabled=disabled):
        try:
            # Ensure env vars (so astrapy picks them up, not strictly required)
            os.environ["ASTRA_DB_API_ENDPOINT"] = astra_endpoint
            os.environ["ASTRA_DB_APP_TOKEN"] = astra_token
            os.environ["ASTRA_DB_KEYSPACE"] = astra_keyspace
            os.environ["ASTRA_COLLECTION"] = astra_collection

            # Prepare documents
            documents: List[Tuple[str, str]] = []  # (source_id, text)

            # Uploaded files
            for f in files or []:
                data = f.read()
                name = f.name.lower()
                if name.endswith(".pdf"):
                    text = _read_pdf_bytes(data)
                else:
                    text = _read_markdown_bytes(data)
                if text.strip():
                    documents.append((f"upload://{f.name}", text))

            # URLs
            if urls_str.strip():
                for u in [u.strip() for u in urls_str.split(",") if u.strip()]:
                    try:
                        text = _fetch_url_text(u)
                        if text.strip():
                            documents.append((u, text))
                    except Exception as ue:
                        st.warning(f"URL fetch failed: {u}: {ue}")

            if not documents:
                st.warning("No valid documents found.")
                st.stop()

            # Chunk
            chunks: List[Tuple[str, str]] = []
            for src, txt in documents:
                for i, ch in enumerate(_chunk_text(txt, max_tokens=max_tokens, overlap=overlap)):
                    chunks.append((f"{src}#chunk-{i}", ch))

            st.info(f"Prepared {len(chunks)} chunks. Getting embeddings from Ollama ({embed_model})…")

            # Probe embedding dim
            probe = _get_ollama_embedding("hello world", base_url=ollama_url, model=embed_model)
            dim = len(probe)

            # Connect Astra and ensure collection
            adb = _connect_astra()
            _ensure_collection(adb, astra_collection, dim)

            # Embed & upsert with a progress bar
            to_upsert = []
            prog = st.progress(0.0, text="Embedding & staging…")
            for idx, (sid, ch) in enumerate(chunks):
                emb = _get_ollama_embedding(ch, base_url=ollama_url, model=embed_model)
                to_upsert.append({
                    "_id": _hash_id(sid),
                    "$vector": emb,
                    "text": ch,
                    "source": sid.split("#chunk-")[0],
                    "meta": {"chunk_id": sid.split("#chunk-")[-1], "ingested_at": int(time.time())}
                })
                if (idx + 1) % 5 == 0 or idx + 1 == len(chunks):
                    prog.progress((idx + 1) / len(chunks))

            _upsert_chunks(adb, astra_collection, to_upsert)
            st.success(f"Ingested {len(to_upsert)} chunks into '{astra_collection}'. You can now use the Chat tab.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
