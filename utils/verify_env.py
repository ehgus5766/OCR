import torch, cv2, numpy
from ultralytics import YOLO
from craft_text_detector import Craft
import easyocr

print(" PyTorch:", torch.__version__)
print(" OpenCV:", cv2.__version__)
print(" Numpy:", numpy.__version__)
print(" YOLO import OK")
print(" EasyOCR import OK")
print("CRAFT import OK")
