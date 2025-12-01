Write-Host "[1/6] PyTorch 및 필수 라이브러리 설치..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

Write-Host "[2/6] 주요 기반 패키지 설치..."
pip install numpy==1.26.4 opencv-python==4.10.0.84 Pillow==10.3.0 scikit-image==0.24.0 matplotlib==3.9.2

Write-Host "[3/6] YOLO, PaddleOCR 설치..."
pip install ultralytics==8.3.20
pip install paddlepaddle==2.6.1
pip install paddleocr==2.7.3

Write-Host "[4/6] OCR 관련 도구 설치..."
pip install easyocr==1.7.2 --no-deps
pip install craft-text-detector==0.4.3 --no-deps

Write-Host "[5/6] Flask, Django, 기타 유틸..."
pip install Flask==3.1.0 Django==4.2.11 tqdm==4.67.1 requests==2.32.3

Write-Host "[6/6] 설치 완료 ✅"
pip list
