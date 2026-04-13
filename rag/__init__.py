"""
RAG Package - Offline AI Knowledge Base System
"""

# Config
from .config import (
    CACHE_DIR, LOCAL_MODEL_DIR, INDEX_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN,
    MAX_FILE_MB, FILE_TIMEOUT_SEC, DEFAULT_TOP_K,
    EMBED_MAX_LEN, BM25_WEIGHT, VECTOR_WEIGHT, KEYWORD_BOOST,
    OLLAMA_HOST, OLLAMA_TIMEOUT,
    WEIGHT_PROFILES, CONFIDENCE_THRESHOLD,
    IGNORE_DIRS, SKIP_EXTENSIONS,
    EMBED_MODELS, DEFAULT_EMBED, RERANKER_HF,
    _embed_st_path, _embed_onnx_path, _reranker_path, _check_available,
    _sanitize_name, _index_base, list_saved_indexes
)

# Utils
from .utils import (
    clean_text, safe_encode,
    _detect_device, _DEVICE, _DEVICE_TYPE, _VRAM_GB,
    _gpu_batch, _gpu_str, _cpu_info, _opt_batch, _opt_threads,
    extract_smart_info
)

# Readers
from .readers import (
    read_txt, read_pdf, read_docx, read_csv, read_xlsx,
    READERS, _file_hash, _read_one, load_documents,
    chunk_text, _chunk_doc, build_chunks_parallel,
    _chunk_hash, EmbeddingCache, FileTracker
)

# Ollama
from .ollama import (
    _find_ollama, get_ollama_models, _ollama_ping, _ollama_call_simple,
    ask_stream, rewrite_query, grade_chunks, semantic_boost_query, multi_step_search
)

# Core
from .core import (
    EmbedEngine, BM25Index, Reranker, HybridStore, _file_relevance_score
)

# Search
from .search import (
    _ABBR, _DOMAIN_EXPAND, expand_query,
    classify_query_type, get_dynamic_weights,
    decompose_query, calculate_confidence_score, enhanced_search_pipeline
)

__all__ = [
    # Config
    "CACHE_DIR", "LOCAL_MODEL_DIR", "INDEX_DIR",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_CHUNK_LEN",
    "MAX_FILE_MB", "FILE_TIMEOUT_SEC", "DEFAULT_TOP_K",
    "EMBED_MAX_LEN", "BM25_WEIGHT", "VECTOR_WEIGHT", "KEYWORD_BOOST",
    "OLLAMA_HOST", "OLLAMA_TIMEOUT",
    "WEIGHT_PROFILES", "CONFIDENCE_THRESHOLD",
    "IGNORE_DIRS", "SKIP_EXTENSIONS",
    "EMBED_MODELS", "DEFAULT_EMBED", "RERANKER_HF",
    "_embed_st_path", "_embed_onnx_path", "_reranker_path", "_check_available",
    "_sanitize_name", "_index_base", "list_saved_indexes",
    # Utils
    "clean_text", "safe_encode",
    "_detect_device", "_DEVICE", "_DEVICE_TYPE", "_VRAM_GB",
    "_gpu_batch", "_gpu_str", "_cpu_info", "_opt_batch", "_opt_threads",
    "extract_smart_info",
    # Readers
    "read_txt", "read_pdf", "read_docx", "read_csv", "read_xlsx",
    "READERS", "_file_hash", "_read_one", "load_documents",
    "chunk_text", "_chunk_doc", "build_chunks_parallel",
    "_chunk_hash", "EmbeddingCache", "FileTracker",
    # Ollama
    "_find_ollama", "get_ollama_models", "_ollama_ping", "_ollama_call_simple",
    "ask_stream", "rewrite_query", "grade_chunks", "semantic_boost_query", "multi_step_search",
    # Core
    "EmbedEngine", "BM25Index", "Reranker", "HybridStore", "_file_relevance_score",
    # Search
    "_ABBR", "_DOMAIN_EXPAND", "expand_query",
    "classify_query_type", "get_dynamic_weights",
    "decompose_query", "calculate_confidence_score", "enhanced_search_pipeline",
]
