"""
setup_check.py — Шаардлагатай бүх зүйлийг шалгаад суулгадаг скрипт
Ажиллуулах: python setup_check.py
"""

import sys
import subprocess
import importlib
import platform
import shutil

# ── Өнгөт текст ──
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✔ {msg}{RESET}")
def fail(msg):  print(f"  {RED}✘ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠ {msg}{RESET}")
def info(msg):  print(f"  {CYAN}→ {msg}{RESET}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}\n" + "─"*50)

# ──────────────────────────────────────────────
# 1. PYTHON ХУВИЛБАР ШАЛГАХ
# ──────────────────────────────────────────────

header("🐍 Python шалгаж байна...")
ver = sys.version_info
print(f"  Хувилбар: Python {ver.major}.{ver.minor}.{ver.micro} ({platform.system()})")

if ver.major < 3 or (ver.major == 3 and ver.minor < 9):
    fail("Python 3.9+ шаардлагатай! Татаж суулгана уу: https://python.org")
    sys.exit(1)
else:
    ok(f"Python {ver.major}.{ver.minor} — тохиромжтой ✓")

# ──────────────────────────────────────────────
# 2. PIP САНГУУД ШАЛГАХ + СУУЛГАХ
# ──────────────────────────────────────────────

PACKAGES = [
    # (import нэр,      pip нэр,                   тайлбар)
    ("gradio",          "gradio>=4.0.0",            "Веб UI"),
    ("sentence_transformers", "sentence-transformers>=2.7.0", "Текст embedding"),
    ("faiss",           "faiss-cpu>=1.7.4",         "Векторын хайлт"),
    ("numpy",           "numpy>=1.24.0",            "Тооцоо"),
    ("pypdf",           "pypdf>=4.0.0",             "PDF уншигч"),
    ("docx",            "python-docx>=1.1.0",       "Word файл уншигч"),
    ("openpyxl",        "openpyxl>=3.1.0",          "Excel файл уншигч"),
    ("pandas",          "pandas>=2.0.0",            "CSV/Excel өгөгдөл"),
    ("ollama",          "ollama>=0.1.0",            "Локал LLM холболт"),
]

header("📦 Python сангуудыг шалгаж байна...")

missing = []
for import_name, pip_name, description in PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "?")
        ok(f"{pip_name.split('>=')[0]:30s} {version:10s}  ({description})")
    except ImportError:
        fail(f"{pip_name.split('>=')[0]:30s} {'суулгаагүй':10s}  ({description})")
        missing.append(pip_name)

# ──────────────────────────────────────────────
# 3. ДУТУУ САНГУУДЫГ СУУЛГАХ
# ──────────────────────────────────────────────

if missing:
    print(f"\n{YELLOW}{BOLD}⚠ {len(missing)} санг суулгах шаардлагатай:{RESET}")
    for m in missing:
        print(f"    • {m}")

    answer = input(f"\n{BOLD}Одоо суулгах уу? [y/n]: {RESET}").strip().lower()
    if answer == "y":
        header("⬇️  Суулгаж байна...")
        for pkg in missing:
            info(f"Суулгаж байна: {pkg}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                ok(f"{pkg.split('>=')[0]} амжилттай суулгагдлаа!")
            else:
                fail(f"{pkg.split('>=')[0]} суулгахад алдаа гарлаа!")
                print(f"    {RED}{result.stderr[:200]}{RESET}")
    else:
        warn("Суулгахаас татгалзлаа. Гараар суулгахын тулд:\n")
        print(f"    pip install {' '.join(missing)}\n")
else:
    print(f"\n  {GREEN}{BOLD}🎉 Бүх Python санг суулгасан байна!{RESET}")

# ──────────────────────────────────────────────
# 4. OLLAMA ШАЛГАХ
# ──────────────────────────────────────────────

header("🤖 Ollama (Локал AI) шалгаж байна...")

ollama_path = shutil.which("ollama")
if ollama_path:
    ok(f"Ollama олдлоо: {ollama_path}")

    # Хувилбар шалгах
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        ok(f"Хувилбар: {r.stdout.strip()}")
    except Exception:
        warn("Хувилбар тодорхойлж чадсангүй")

    # Суулгасан загваруудыг харах
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        lines = r.stdout.strip().splitlines()
        if len(lines) > 1:
            ok(f"{len(lines)-1} загвар суулгасан байна:")
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    print(f"    {GREEN}•{RESET} {parts[0]}")
        else:
            warn("Одоогоор загвар суулгаагүй байна")
            print(f"\n  {CYAN}Загвар татах командууд:{RESET}")
            print(f"    ollama pull qwen2.5   {GREEN}← Санал болгоно (Монгол/Англи){RESET}")
            print(f"    ollama pull llama3.2  {CYAN}← Хөнгөн (2GB){RESET}")
            print(f"    ollama pull mistral   {CYAN}← Сайн чанар (4GB){RESET}")

            answer = input(f"\n{BOLD}qwen2.5 загварыг одоо татах уу? [y/n]: {RESET}").strip().lower()
            if answer == "y":
                info("Татаж байна... (хэдэн минут болж болно)")
                subprocess.run(["ollama", "pull", "qwen2.5"])
                ok("qwen2.5 амжилттай татагдлаа!")
    except Exception as e:
        warn(f"Ollama серверт холбогдож чадсангүй: {e}")
        info("'ollama serve' командыг өөр терминалд ажиллуулна уу")

else:
    fail("Ollama суулгаагүй байна!")
    print(f"""
  {YELLOW}Ollama суулгах заавар:{RESET}

  Windows:
    1. https://ollama.ai хаягаас татна
    2. OllamaSetup.exe ажиллуулна

  Mac:
    brew install ollama
    эсвэл https://ollama.ai-аас татна

  Linux:
    curl -fsSL https://ollama.ai/install.sh | sh
""")

# ──────────────────────────────────────────────
# 5. ДҮГНЭЛТ
# ──────────────────────────────────────────────

header("📋 Дүгнэлт")

# Дахин шалгах
all_ok = True
for import_name, pip_name, _ in PACKAGES:
    try:
        importlib.import_module(import_name)
    except ImportError:
        all_ok = False
        break

if all_ok and ollama_path:
    print(f"  {GREEN}{BOLD}✅ Бүх зүйл бэлэн! Системийг эхлүүлж болно:{RESET}")
    print(f"\n    {BOLD}python app.py{RESET}\n")
elif all_ok:
    print(f"  {YELLOW}{BOLD}⚠ Python санг бэлэн, гэхдээ Ollama суулгаагүй байна.{RESET}")
    print(f"  Ollama суулгасны дараа: {BOLD}python app.py{RESET}\n")
else:
    print(f"  {RED}{BOLD}❌ Зарим санг суулгаагүй байна. Дахин ажиллуулна уу:{RESET}")
    print(f"\n    {BOLD}python setup_check.py{RESET}\n")