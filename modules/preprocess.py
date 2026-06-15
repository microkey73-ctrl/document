import cv2

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    return gray, edge

이미지를 문서 찾기 전에 깔끔하게 전처리 하는 함수

예를 들어 
