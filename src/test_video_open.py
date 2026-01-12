import cv2

path = "videos/traffic_sign_day_cv.mp4"
cap = cv2.VideoCapture(path)

ok, frame = cap.read()
cap.release()

print("Opened:", ok)
if ok:
    print("Frame shape:", frame.shape)
