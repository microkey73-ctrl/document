import cv2
from modules.preprocess import preprocess
from modules.detector import find_document_contour
from modules.transform import warp_perspective, get_scanned_effect

# 카메라 연결
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 자동 초점 활성화

# 이전 프레임의 문서 좌표를 저장할 변수 (깜빡임 방지용 캐시)
last_doc_contour = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라를 불러올 수 없습니다.")
        break

    # 원본 프레임 복사본 생성 (초록색 선이 없는 순수한 화면을 스캔하기 위함)
    display_frame = frame.copy()

    # 1. 전처리
    gray, edge = preprocess(frame)

    import cv2
    from modules.preprocess import preprocess
    from modules.detector import find_document_contour
    from modules.transform import warp_perspective, get_scanned_effect

    # 카메라 연결 (해상도 설정 삭제 -> 원래 해상도로 복구)
    cap = cv2.VideoCapture(0)

    # 이전 프레임의 문서 좌표를 저장할 변수 (깜빡임 방지용 캐시)
    last_doc_contour = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라를 불러올 수 없습니다.")
            break

        # 원본 프레임 복사본 생성 (초록색 선이 없는 순수한 화면을 스캔하기 위함)
        display_frame = frame.copy()

        # 1. 전처리
        gray, edge = preprocess(frame)

        # 2. 문서 검출
        doc = find_document_contour(edge)

        # 3. 문서 좌표 업데이트 및 유지
        if doc is not None:
            last_doc_contour = doc  # 새로운 문서가 검출되면 업데이트

        # 4. 스캔 결과 출력 (기억해둔 좌표가 있다면 그것으로 출력)
        if last_doc_contour is not None:
            try:
                # 초록색 선이 그려지지 않은 원본(frame)으로 워핑 진행
                scanned = warp_perspective(frame, last_doc_contour)
                scanned_effect = get_scanned_effect(scanned)

                cv2.imshow("Scanned", scanned_effect)
            except Exception as e:
                # 변환 오류 발생 시 캐시 초기화
                last_doc_contour = None

            # 디스플레이용 프레임에만 초록색 사각형 그리기
            cv2.drawContours(display_frame, [last_doc_contour], -1, (0, 255, 0), 2)
        else:
            # 검출된 적이 없다면 안내 메시지 레이블 표시
            cv2.putText(display_frame, "Bring the document closer", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Scanned", display_frame)

        # 원본 카메라 화면 표시 (초록선이나 메시지가 포함된 화면)
        cv2.imshow("Camera", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()