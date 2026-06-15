import cv2
import numpy as np
from modules.utils import order_points


def warp_perspective(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 가로, 세로 길이 계산 (유클리드 거리)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # [개선 3] 문서 가장자리 글씨가 잘리는 것을 방지하기 위해 사방에 10픽셀 마진 부여
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
    """ 영수증/문서 스캔 느낌을 내는 전처리 """
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

    # [개선 4] 이진화 전 미세한 가우시안 블러로 글씨 외곽선 노이즈 제거
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # [개선 5] BlockSize를 25로 늘리고, C값을 5로 낮춰서 글씨가 파묻히지 않고 두껍게 나오도록 수정
    scanned = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        5
    )
    return scanned