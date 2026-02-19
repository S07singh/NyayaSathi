"""
NyayaSathi AI — Centralised Configuration
==========================================
All tuneable parameters live here so that no magic strings or numbers
are scattered across the codebase.  Values that depend on the runtime
environment (API keys, Ollama host) are read from environment variables
with sensible defaults.
"""

import os


# ──────────────────────────── LLM Settings ────────────────────────────

# Groq (Fast Mode)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.1-70b-versatile"
GROQ_TIMEOUT: int = 30          # seconds

# Ollama (Private Mode)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = "llama3.1"
OLLAMA_TIMEOUT: int = 120       # local inference can be slower

# Shared generation defaults
DEFAULT_TEMPERATURE: float = 0.3
DEFAULT_MAX_TOKENS: int = 2048


# ──────────────────────────── Embedding Settings ──────────────────────

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384   # output dim of all-MiniLM-L6-v2


# ──────────────────────────── RAG / FAISS Settings ────────────────────

FAISS_INDEX_DIR: str = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
LEGAL_REFERENCES_DIR: str = os.path.join(os.path.dirname(__file__), "data", "legal_references")
CHUNK_SIZE: int = 512            # characters per chunk
CHUNK_OVERLAP: int = 64          # overlap between consecutive chunks
TOP_K: int = 5                   # number of retrieved chunks


# ──────────────────────────── Risk Scoring Weights ────────────────────

RISK_WEIGHTS: dict[str, float] = {
    "penalty_severity":       0.30,
    "clause_imbalance":       0.25,
    "termination_unfairness": 0.20,
    "liability_excess":       0.15,
    "missing_rights":         0.10,
}


# ──────────────────────────── Document Categories ─────────────────────

DOCUMENT_CATEGORIES: list[str] = [
    "Rental Agreement",
    "Employment Contract",
    "Small Business Contract",
    "Legal Notice",
]


# ──────────────────────────── Logging ─────────────────────────────────

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ──────────────────────────── App Meta ────────────────────────────────

APP_TITLE: str = "NyayaSathi AI"
APP_SUBTITLE: str = "Legal Document Simplifier & Risk Analyzer"
APP_VERSION: str = "1.0.0"
