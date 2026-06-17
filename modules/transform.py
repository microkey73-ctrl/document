import cv2
import numpy as np
from modules.utils import order_points


def warp_perspective(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    margin = 10
    dst = np.array([
        [margin, margin],
        [maxWidth - 1 - margin, margin],
        [maxWidth - 1 - margin, maxHeight - 1 - margin],
        [margin, maxHeight - 1 - margin]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped


def get_scanned_effect(warped_img):

    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    scanned = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        5
    )
    return scanned

문서를 스캔처럼 만드는 파일
무선 펴기 + 스캔 효과 
기울어진 문서를 정면으로 펴고 흑백 + 이진화 해서 스캔 느낌 살리기 
