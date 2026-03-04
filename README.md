# 🧩 Multi‑Agent RAG (Langflow + Streamlit + Astra DB) — Free Tier Setup
Link for direct access : https://multi-agent-app-t2sygtgjvehg8uygmhn2zb.streamlit.app/

This template recreates the Tech With Tim multi‑agent RAG idea using **Langflow** for agent graphs, **Streamlit** for UI, and **Astra DB** for vector storage. It is designed to run **for ₹0** using:
- Streamlit Community Cloud (hosted URL)
- Astra DB Free plan
- Free/open LLMs via **Ollama** (self‑hosted) or any free LLM endpoint

## 1) Langflow
- Install locally: `pip install langflow`
- Run: `langflow run --host 0.0.0.0 --port 7860`
- Import `flows/multi_agent_rag.json`
- In the graph, set:
  - **Astra DB** credentials (endpoint, token, keyspace, collection)
  - **Embeddings** model (e.g., `all-minilm` via Ollama or a hosted embedding API)
  - **LLM** provider (Ollama `llama3:8b` or your choice)
- Get your **Flow ID** (in Flow settings) and create an API key in Langflow settings.

## 2) Astra DB (Free)
- Create a Serverless DB at https://astra.datastax.com
- Get `API Endpoint` and `App Token`
- Optional: run `scripts/init_astra.py` to precreate a vector collection:
  ```bash
  pip install -r requirements.txt
  export LANGFLOW_* ASTRA_*  # or set in .env / Secrets
  python scripts/init_astra.py
  ```

## 3) Streamlit (local test)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # edit values
streamlit run app/streamlit_app.py
```

## 4) Deploy to Streamlit Community Cloud (FREE URL)
1. Push this repo to GitHub (public).
2. In Streamlit Cloud → "Deploy an app" → select repo/branch → main file: `app/streamlit_app.py`.
3. Add **Secrets**:
   ```toml
   LANGFLOW_BASE_URL = "https://<your-langflow-host>"  # can be your home IP:port if exposed via tunnel
   LANGFLOW_API_KEY = "<your-key>"
   LANGFLOW_FLOW_ID = "multi-agent-rag"

   # if you will run scripts/init_astra.py via local machine, these are optional in Cloud
   ASTRA_DB_API_ENDPOINT = "https://<db-id>-<region>.apps.astra.datastax.com"
   ASTRA_DB_APP_TOKEN = "astraCS:..."
   ASTRA_DB_KEYSPACE = "default_keyspace"
   ASTRA_COLLECTION = "docs"
   ```
4. Click **Deploy** → you get a public URL.

### Notes
- **Do not run heavy models on Streamlit Cloud.** Host LLMs elsewhere (Ollama on your PC/VPS). Streamlit only sends API calls.
- **Astra Free** may hibernate when idle; first query might be slower. Implement retries in production.
- Keep keys in **Secrets** or env vars, not in code.

## 5) Ingesting your docs
- **Inside Langflow**: add a loader (Web/PDF), pipe → Embeddings → AstraDB VectorStore.
- **Outside (Python)**: write a script to embed + upsert via Data API (see `scripts/init_astra.py` start).

## Troubleshooting
- Streamlit shows `(No text returned)` → ensure your Flow ends with a **ChatOutput** and the API call uses `output_type=chat`.
- 401 from Langflow → create & include `x-api-key`.
- Vector dimension mismatch → set Astra collection dimension to your embedding model size (e.g., 384 for MiniLM).


---
## 6) Ingest documents (PDF/URL/Markdown) → Astra DB
**Requirements:** an embeddings endpoint (defaults to Ollama at `http://localhost:11434`) and Astra DB credentials set in env or `secrets.toml`.

**Environment** (examples):
```bash
export ASTRA_DB_API_ENDPOINT="https://<db-id>-<region>.apps.astra.datastax.com"
export ASTRA_DB_APP_TOKEN="astraCS:..."
export ASTRA_DB_KEYSPACE="default_keyspace"
export ASTRA_COLLECTION="docs"

export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_EMBED_MODEL="all-minilm:latest"
```

**Run ingestion:**
```bash
# PDFs from a folder
python scripts/ingest.py --pdf_dir ./my_pdfs

# Web pages (space separated)
python scripts/ingest.py --urls https://example.com https://docs.python.org

# Markdown/notes
python scripts/ingest.py --md_dir ./notes

# Combine sources
python scripts/ingest.py --pdf_dir ./my_pdfs --urls https://example.com --md_dir ./notes
```

This will chunk text, create embeddings via **Ollama**, and upsert vectors + metadata into the Astra collection specified by `ASTRA_COLLECTION`.
Ensure your **Langflow Retriever** points to the same collection + dimension.
