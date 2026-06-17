import numpy as np

def order_points(pts):

    pts = pts.reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_side = x_sorted[:2, :]
    right_side = x_sorted[2:, :]

    tl = left_side[np.argmin(left_side[:, 1])]
    bl = left_side[np.argmax(left_side[:, 1])]

    tr = right_side[np.argmin(right_side[:, 1])]
    br = right_side[np.argmax(right_side[:, 1])]

    return np.array([tl, tr, br, bl], dtype="float32")
