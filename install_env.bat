@echo off
echo [1/5] Python 3.10 가상환경 생성 중...
rmdir /s /q .venv >nul 2>&1
python -m venv .venv

echo [2/5] 가상환경 활성화 및 pip 최신화...
call .\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel cmake scikit-build ninja

echo [3/5] PyTorch 및 핵심 의존성 설치...
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 opencv-python==4.10.0.84 Pillow==10.3.0 scikit-image==0.24.0 matplotlib==3.9.2

echo [4/5] 주요 AI 프레임워크 설치...
pip install ultralytics==8.3.20
pip install paddlepaddle==2.6.1
pip install paddleocr==2.7.3

echo [5/5] OCR / Detection 보조 패키지 설치...
pip install easyocr==1.7.2 --no-deps
pip install craft-text-detector==0.4.3 --no-deps

echo 환경 세팅 완료
pause
