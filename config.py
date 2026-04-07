"""Configuração centralizada — carrega .env e define caminhos."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """Busca config: primeiro st.secrets (Streamlit Cloud), depois os.environ/.env."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

# ── Diretórios ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / _get("DATA_DIR", "data")
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHROMADB_DIR = DATA_DIR / "chromadb"
EPISODES_FILE = DATA_DIR / "episodes.json"

for d in [AUDIO_DIR, TRANSCRIPTS_DIR, CHROMADB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Podcast Index ───────────────────────────────────────────
PI_API_KEY = _get("PODCAST_INDEX_API_KEY", "")
PI_API_SECRET = _get("PODCAST_INDEX_API_SECRET", "")
PI_FEED_ID = _get("PODCAST_INDEX_FEED_ID", "2788697")
PI_BASE_URL = "https://api.podcastindex.org/api/1.0"

# ── Azure OpenAI ────────────────────────────────────────────
AZURE_ENDPOINT = _get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = _get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = _get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
AZURE_TRANSCRIPTION_MODEL = _get("AZURE_TRANSCRIPTION_DEPLOYMENT", "gpt-4o-transcribe-2")
AZURE_EMBEDDING_MODEL = _get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_EMBEDDING_API_VERSION = _get("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
AZURE_DIARIZE_MODEL = _get("AZURE_DIARIZE_DEPLOYMENT", "gpt-4o-transcribe-diarize-2")

# ── Azure Speech (diarização) ───────────────────────────────
AZURE_SPEECH_KEY = _get("AZURE_SPEECH_KEY", AZURE_API_KEY)
AZURE_SPEECH_REGION = _get("AZURE_SPEECH_REGION", "eastus2")

# ── Azure Chat (GPT-4o para extração) ──────────────────────
AZURE_CHAT_MODEL = _get("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_CHAT_API_VERSION = _get("AZURE_CHAT_API_VERSION", "2025-03-01-preview")

# ── Firecrawl ───────────────────────────────────────────────
FIRECRAWL_API_KEY = _get("FIRECRAWL_API_KEY", "")
FIRECRAWL_BASE_URL = _get("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")

# ── Knowledge Graph ────────────────────────────────────────
KG_DIR = DATA_DIR / "knowledge_graph"
KG_ENTITIES_DIR = KG_DIR / "entities"
KG_GRAPH_FILE = KG_DIR / "graph.json"
KG_NETWORKX_FILE = KG_DIR / "graph.graphml"

for d in [KG_DIR, KG_ENTITIES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

KG_CHUNK_SIZE = 3000
KG_CHUNK_OVERLAP = 200
KG_MAX_RETRIES = 3
KG_BATCH_DELAY = 0.5

# ── RAG ─────────────────────────────────────────────────────
CHUNK_SIZE = 500        # tokens por chunk
CHUNK_OVERLAP = 50      # tokens de overlap
EMBEDDING_BATCH = 100   # textos por request de embedding
MAX_RESULTS = 8         # chunks retornados na busca
