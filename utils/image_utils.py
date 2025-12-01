import cv2
import numpy as np

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    #감마값 조정
    if brightness > 180:
        gamma = 1.3
    elif brightness < 70:
        gamma = 0.9
    else:
        gamma = 1.0
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    #대비
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr , cb =  cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    ycrcb = cv2.merge([y, cr, cb])
    enhanced = cv2. cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    #대비 보정
    enhanced = cv2.bilateralFilter(enhanced, 5, 80, 80)
    #
    #enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
    blur = cv2.GaussianBlur(enhanced, (3,3), 0)
    enhanced = cv2.addWeighted(enhanced, 1.8, blur, -0.8, 0)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * 0.9, 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 4
    )
    enhanced = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

    if enhanced.shape[1] < 2000:
        scale = 2000 / enhanced.shape[1]
        enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return enhanced