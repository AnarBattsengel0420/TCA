@echo off
REM ══════════════════════════════════════════════════════════
REM  Intel UHD 620 хурдасгалтын тохиргоо
REM  Ажиллуулах: setup_intel.bat
REM ══════════════════════════════════════════════════════════

echo.
echo ============================================
echo  Intel UHD 620 RAG хурдасгалт
echo ============================================
echo.

REM 1. OpenVINO суулгах (Intel CPU/GPU хурдасгал)
echo [1/4] OpenVINO суулгаж байна...
pip install optimum[onnxruntime] onnxruntime-openvino --break-system-packages 2>nul || pip install optimum[onnxruntime] onnxruntime-openvino

REM 2. Хуучин кэш цэвэрлэх
echo.
echo [2/4] Хуучин кэш цэвэрлэж байна...
python App.py --clear-cache

REM 3. ONNX model дахин татах (OpenVINO-д тохируулсан)
echo.
echo [3/4] ONNX model бэлдэж байна...
python App.py --download

REM 4. Бэлэн
echo.
echo ============================================
echo  Бэлэн! Ажиллуулах:
echo    ollama serve
echo    python App.py
echo.
echo  Хүлээгдэж буй хурдасгал:
echo    Embedding: ~2-4x хурдан (OpenVINO)
echo    Query:     ~10-20x хурдан (LLM cache)
echo    Indexing:  ~2x хурдан (кэш цэвэр)
echo ============================================
pause