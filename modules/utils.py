import numpy as np

def order_points(pts):
    # 4개의 좌표를 x축 기준으로 정렬
    pts = pts.reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # x 좌표가 작은 2개(좌측), 큰 2개(우측) 분리
    left_side = x_sorted[:2, :]
    right_side = x_sorted[2:, :]

    # 좌측 2개 중 y가 작은 것이 좌상(tl), 큰 것이 좌하(bl)
    tl = left_side[np.argmin(left_side[:, 1])]
    bl = left_side[np.argmax(left_side[:, 1])]

    # 우측 2개 중 y가 작은 것이 우상(tr), 큰 것이 우하(br)
    tr = right_side[np.argmin(right_side[:, 1])]
    br = right_side[np.argmax(right_side[:, 1])]

    return np.array([tl, tr, br, bl], dtype="float32")