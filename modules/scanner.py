import cv2
import numpy as np

def order_points(pts):
 4개의 좌표를 [좌상, 우상, 우하, 좌하] 순서로 정렬
    pts = pts.reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_side = x_sorted[:2, :]
    right_side = x_sorted[2:, :]

    tl = left_side[np.argmin(left_side[:, 1])]
    bl = left_side[np.argmax(left_side[:, 1])]

    tr = right_side[np.argmin(right_side[:, 1])]
    br = right_side[np.argmax(right_side[:, 1])]

    return np.array([tl, tr, br, bl], dtype="float32")


def warp_perspective(img, pts):
정렬된 좌표를 기준으로 이미지 원근 변환 (기울기 보정)
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
 영수증/문서 스캔 느낌을 내는 이진화 처리
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


def preprocess(img):
에지(테두리) 검출을 위한 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    return gray, edge


def find_document_contour(edge):
 이미지에서 가장 큰 사각형 윤곽선 탐색 
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx
    return None

def run_scanner():
    cap = cv2.VideoCapture(0)  # 0번 기본 카메라 구동

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("=========================================")
    print(" [문서 스캔 자동 보정 프로그램 작동 중] ")
    print(" - Spacebar : 현재 문서 스캔 및 고정")
    print(" - Q        : 프로그램 종료")
    print("=========================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져오지 못했습니다.")
            break

        display_frame = frame.copy()

        gray, edge = preprocess(frame)
        doc_contour = find_document_contour(edge)

        if doc_contour is not None:
            cv2.drawContours(display_frame, [doc_contour], -1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Ready to Scan (Press SPACE)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Scanning for document...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Real-time Scanner", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if doc_contour is not None:
                print("[알림] 보정 작업을 진행합니다...")
                pts = doc_contour.reshape(4, 2)
                warped = warp_perspective(frame, pts)

                scanned_result = get_scanned_effect(warped)

                cv2.imshow("Scanned Result", scanned_result)

                print(" -> [확인 완료] 아무 키나 누르면 카메라 화면으로 돌아갑니다.")
                cv2.waitKey(0)
                cv2.destroyWindow("Scanned Result")
            else:
                print("[경고] 문서 윤곽선이 명확하지 않습니다. 초록색 선이 생겼을 때 눌러주세요.")

        elif key == ord('q'):
            print("[알림] 프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================================================================
# [3] 파이썬 스크립트가 실행될 때 메인 함수를 호출하는 부분
# =========================================================================
if __name__ == "__main__":
    run_scanner()
