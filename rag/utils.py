"""
RAG Utilities - Hardware detection, Text cleaning, Pattern matching
"""
import os
import re


# ── Текст цэвэрлэгч ─────────────────────────────────────────
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff]")

def clean_text(t):
    """Control character-ууд болон invalid unicode устгах"""
    return _RE_CTRL.sub("", t) if t else ""

def safe_encode(t):
    """UTF-8 болгон хөрвүүлэх (алдааг орлуулж)"""
    return clean_text(t).encode("utf-8", errors="replace")


# ── Safe print (Windows console encoding) ────────────────────
def _safe_print(msg):
    """Windows console дээр unicode алдаа гарахаас сэргийлнэ"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


# ── GPU илрүүлэлт ───────────────────────────────────────────
def _detect_device():
    """GPU/CPU илрүүлж device, type, VRAM буцаана"""
    try:
        import torch
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _safe_print(f"  GPU: {name} ({total:.1f}GB VRAM)")
            return torch.device("cuda"), "cuda", total
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _safe_print("  GPU: Apple MPS")
            return torch.device("mps"), "mps", 0.0
        ver = getattr(torch.version, "cuda", None)
        msg = f"CPU-only ({torch.__version__})" if not ver else f"CUDA={ver} (GPU not found)"
        _safe_print(f"  {msg}")
    except ImportError:
        _safe_print("  torch not installed")
    _safe_print("  Running in CPU mode")
    return None, "cpu", 0.0

# Device мэдээлэл глобал
_DEVICE, _DEVICE_TYPE, _VRAM_GB = _detect_device()


def _gpu_batch(bs):
    """GPU-д тохирох batch size тооцоолох"""
    if _DEVICE_TYPE == "cpu":  return bs
    if _DEVICE_TYPE == "mps":  return min(bs*4, 512)
    if _VRAM_GB >= 16:         return min(bs*8, 1024)
    if _VRAM_GB >= 8:          return min(bs*4, 512)
    if _VRAM_GB >= 4:          return min(bs*2, 256)
    return bs


def _gpu_str():
    """GPU мэдээллийг string болгох"""
    if _DEVICE_TYPE == "cuda": return f"CUDA ({_VRAM_GB:.1f}GB)"
    if _DEVICE_TYPE == "mps":  return "Apple MPS"
    return "CPU"


def _cpu_info():
    """CPU core, RAM мэдээлэл авах"""
    cores = os.cpu_count() or 4
    try:
        import psutil
        ram = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram = 8.0
    return cores, ram


def _opt_batch():
    """RAM-д тохирох batch size тооцоолох"""
    _, ram = _cpu_info()
    if ram < 8:  return 8
    if ram < 16: return 16
    if ram < 32: return 32
    return 64


def _opt_threads():
    """Оновчтой thread тоо тооцоолох"""
    cores, _ = _cpu_info()
    return max(1, min(cores//2, 4))


# ── Pattern matching ───────────────────────────────────────
def extract_smart_info(content, query):
    """Query-тэй холбоотой key-value, matching lines олох"""
    extracted = []
    q_lower   = query.lower()
    q_words   = {w for w in re.findall(r'\w+', q_lower, re.UNICODE) if len(w) >= 2}

    # Key-value patterns
    for pat in [r'([\w\s]{2,30})\s*[:=]\s*([^\n\r]{3,100})',
                r'([\w\s]{2,30})\s*[-–—]\s*([^\n\r]{3,100})']:
        for key, val in re.findall(pat, content):
            kc = key.strip().lower()
            kw = set(re.findall(r'\w+', kc, re.UNICODE))
            if q_words & kw or any(qw in kc for qw in q_words if len(qw) > 2):
                vc = val.strip()
                if 2 < len(vc) < 200:
                    extracted.append(f"✓ {key.strip()}: {vc}")

    # Matching lines
    cl = content.lower()
    for word in q_words:
        if len(word) < 3:
            continue
        m = re.search(re.escape(word), cl)
        if m:
            s = max(0, m.start() - 100)
            e = min(len(content), m.end() + 150)
            for line in content[s:e].split('\n'):
                if word in line.lower() and 10 < len(line.strip()) < 200 and len(extracted) < 5:
                    extracted.append(f"→ {line.strip()}")
            break

    # Давхардал арилгах
    seen, unique = set(), []
    for item in extracted:
        k = re.sub(r'[^\w\s]', '', item.lower())
        if k not in seen and len(k) > 5:
            seen.add(k)
            unique.append(item)
    return unique[:6]
