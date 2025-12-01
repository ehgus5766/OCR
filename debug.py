import torch

ckpt = torch.load("C:/Users/user/Desktop/CV/ocr/weights/crnn_best.pth", map_location="cpu")

print("Keys:", ckpt.keys())

if "alphabet" in ckpt:
    print("Alphabet length:", len(ckpt["alphabet"]))
    print("Sample:", ckpt["alphabet"][:100])
else:
    print("⚠️ alphabet key 없음")