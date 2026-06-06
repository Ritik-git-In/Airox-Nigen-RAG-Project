"""Central configuration, loaded from environment / .env file.

Importing this module loads the .env file once and exposes typed settings the
rest of the app reads. Keeping it in one place means there is a single source of
truth for paths, model names, and tuning knobs.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the parent of this file's package directory).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _setting(key: str, default: str = "") -> str:
    """Read a setting from the environment (.env locally) and, if missing, from
    Streamlit's secrets — so the same code works on Streamlit Community Cloud,
    where secrets are provided via the dashboard instead of a .env file.
    """
    val = os.getenv(key)
    if val:
        return val
    try:  # st.secrets only exists when running under Streamlit
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# Where uploaded PDFs and the vector database live. Created on first use.
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

# ---- Kimi / Moonshot (generation) ----
KIMI_API_KEY = _setting("KIMI_API_KEY", "")
KIMI_BASE_URL = _setting("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = _setting("KIMI_MODEL", "moonshot-v1-32k")

# ---- Local embeddings ----
EMBEDDING_MODEL = _setting("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---- Chunking / retrieval ----
CHUNK_SIZE = int(_setting("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_setting("CHUNK_OVERLAP", "150"))
TOP_K = int(_setting("TOP_K", "5"))


def ensure_dirs() -> None:
    """Create the runtime data directories if they don't exist yet."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def kimi_is_configured() -> bool:
    """True if a Kimi API key is present (so the UI can warn early if not)."""
    return bool(KIMI_API_KEY and KIMI_API_KEY != "sk-your-kimi-key-here")
