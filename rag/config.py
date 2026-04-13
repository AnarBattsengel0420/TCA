"""
RAG Configuration - Бүх тохиргоо
"""
import os
import re
from pathlib import Path

# ── Үндсэн тохиргоо ──────────────────────────────────────────
CACHE_DIR        = Path.home() / ".rag_cache"
LOCAL_MODEL_DIR  = CACHE_DIR / "local_models"
INDEX_DIR        = CACHE_DIR / "indexes"

CHUNK_SIZE       = 900
CHUNK_OVERLAP    = 150
MIN_CHUNK_LEN    = 10
MAX_FILE_MB      = 50
FILE_TIMEOUT_SEC = 30
DEFAULT_TOP_K    = 5
EMBED_MAX_LEN    = 512
BM25_WEIGHT      = 0.70
VECTOR_WEIGHT    = 0.30
KEYWORD_BOOST    = 0.70
OLLAMA_HOST      = "http://localhost:11434"
OLLAMA_TIMEOUT   = 600

# ── Enhanced Search Тохиргоо ─────────────────────────────────
# Exact match bonus - үг яг таарсан бол нэмэлт оноо
EXACT_MATCH_BOOST = 0.35      # Бүтэн query таарсан
EXACT_WORD_BOOST  = 0.15      # Үг бүр яг таарсан (word boundary)

WEIGHT_PROFILES = {
    "factual":    {"bm25": 0.55, "vector": 0.25, "keyword": 0.20},
    "semantic":   {"bm25": 0.30, "vector": 0.55, "keyword": 0.15},
    "exact":      {"bm25": 0.30, "vector": 0.10, "keyword": 0.60},
    "mixed":      {"bm25": 0.45, "vector": 0.30, "keyword": 0.25},
}
CONFIDENCE_THRESHOLD = 0.65

# ── Файл шүүлтүүр ────────────────────────────────────────────
IGNORE_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__"}

SKIP_EXTENSIONS = {
    ".py",".js",".ts",".jsx",".tsx",".java",".c",".cpp",".h",".cs",
    ".go",".rs",".rb",".php",".swift",".kt",".sh",".bat",".ps1",".lua",
    ".json",".yaml",".yml",".toml",".ini",".cfg",".env",".xml",".config",
    ".html",".htm",".css",".scss",
    ".png",".jpg",".jpeg",".gif",".bmp",".svg",".ico",".webp",
    ".mp4",".mp3",".avi",".mov",".wav",".zip",".tar",".gz",".rar",".7z",
    ".exe",".dll",".so",".db",".sqlite",".pkl",".bin",".npy",".lock",
}

# ── Embedding загварууд ──────────────────────────────────────
EMBED_MODELS = {
    "multilingual-fast": {
        "hf_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384, "desc": "Олон хэл хурдан (470MB) — Монгол", "multilingual": True,
    },
    "multilingual-quality": {
        "hf_name": "intfloat/multilingual-e5-small",
        "dim": 384, "desc": "Олон хэл чанартай (470MB) — Монгол",
        "multilingual": True, "use_prefix": True,
    },
    "ultra-fast": {
        "hf_name": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "dim": 384, "desc": "Хамгийн хурдан (17MB) — зөвхөн Англи", "multilingual": False,
    },
    "fast": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384, "desc": "Тэнцвэртэй (80MB) — зөвхөн Англи", "multilingual": False,
    },
}
DEFAULT_EMBED = "multilingual-fast"
RERANKER_HF   = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Path functions ───────────────────────────────────────────
def _embed_st_path(key):
    return LOCAL_MODEL_DIR / EMBED_MODELS[key]["hf_name"].replace("/","_")

def _embed_onnx_path(key):
    return LOCAL_MODEL_DIR / (EMBED_MODELS[key]["hf_name"].replace("/","_") + "_onnx")

def _reranker_path():
    return LOCAL_MODEL_DIR / RERANKER_HF.replace("/","_")

def _check_available():
    return {k: _embed_st_path(k).exists() for k in EMBED_MODELS}


# ── Index functions ──────────────────────────────────────────
def _sanitize_name(name):
    name = re.sub(r'[\\/:*?"<>|]', "_", name.strip())
    return re.sub(r'\s+', "_", name)[:80] or "default"

def _index_base(name):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return str(INDEX_DIR / _sanitize_name(name))

def list_saved_indexes():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return [f.stem for f in sorted(INDEX_DIR.glob("*.faiss"),
            key=lambda x: x.stat().st_mtime, reverse=True)]
