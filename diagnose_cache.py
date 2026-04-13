"""
🔬 RAG Cache оношлогоо + цэвэрлэгч
Ажиллуулах: python diagnose_cache.py

Юу хийх вэ:
1. Кэш файлуудын хэмжээг шалгана
2. Embedding кэш хэр том болсныг харуулна
3. Tracker файлуудыг шалгана
4. Удаан байгаа шалтгааныг тодорхойлно

Кэш цэвэрлэхдээ: python App.py --clear-cache
"""
import os, pickle
from pathlib import Path

CACHE_DIR = Path.home() / ".rag_cache"

def human_size(b):
    for u in ["B","KB","MB","GB"]:
        if b < 1024: return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}TB"

print("=" * 60)
print("🔬 RAG Кэш оношлогоо")
print("=" * 60)

# 1. Total size
total = 0
for f in CACHE_DIR.rglob("*"):
    if f.is_file():
        total += f.stat().st_size
print(f"\n📁 Нийт кэш: {human_size(total)}")

# 2. Embedding cache
emb_dir = CACHE_DIR / "embeddings"
if emb_dir.exists():
    for f in emb_dir.glob("*.pkl"):
        sz = f.stat().st_size
        print(f"\n💎 Embedding кэш: {f.name} — {human_size(sz)}")
        if sz > 500_000_000:
            print(f"   ⚠️  ЭНЭ МАААШ ТОМ! Энэ нь удаашралын гол шалтгаан байж магадгүй.")
            print(f"   → python App.py --clear-cache")
        try:
            with open(f, "rb") as fh:
                store = pickle.load(fh)
            print(f"   Бичлэг: {len(store):,}")
            if len(store) > 100000:
                print(f"   ⚠️  100K+ бичлэг — кэш хэт их өссөн")
        except Exception as e:
            print(f"   ❌ Уншиж чадсангүй: {e}")
            print(f"   → Эвдэрсэн байж магадгүй. Устга: python App.py --clear-cache")

# 3. Trackers
trk_dir = CACHE_DIR / "trackers"
if trk_dir.exists():
    trk_files = list(trk_dir.glob("*.pkl"))
    print(f"\n📋 Tracker файл: {len(trk_files)}")
    for f in trk_files:
        try:
            with open(f, "rb") as fh:
                h = pickle.load(fh)
            print(f"   {f.name}: {len(h)} файл tracked")
        except Exception:
            print(f"   {f.name}: ❌ эвдэрсэн")

# 4. Chunk stores
cs_dir = CACHE_DIR / "chunk_stores"
if cs_dir.exists():
    cs_files = list(cs_dir.glob("*.pkl"))
    print(f"\n📦 Chunk store: {len(cs_files)}")
    for f in cs_files:
        sz = f.stat().st_size
        print(f"   {f.name}: {human_size(sz)}")
        if sz > 200_000_000:
            print(f"   ⚠️  Хэт том chunk store")

# 5. Indexes
idx_dir = CACHE_DIR / "indexes"
if idx_dir.exists():
    for f in sorted(idx_dir.glob("*.faiss")):
        meta_f = f.with_suffix(".meta")
        sz = f.stat().st_size
        info = ""
        if meta_f.exists():
            try:
                with open(meta_f, "rb") as fh:
                    meta = pickle.load(fh)
                n = len(meta.get("chunks", []))
                info = f" | {n:,} chunk"
            except Exception:
                pass
        print(f"\n🗂️  {f.stem}: {human_size(sz)}{info}")

print(f"\n{'=' * 60}")
print("💡 Хэрэв удаан байвал:")
print("   1. python App.py --clear-cache  (кэш бүрэн цэвэрлэх)")
print("   2. Дахин индексэлнэ")
print("   3. Шинэ .venv үүсгэх шаардлагагүй — кэш цэвэрлэхэд хангалттай")
print(f"{'=' * 60}")