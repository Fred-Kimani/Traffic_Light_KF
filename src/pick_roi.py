import cv2

VIDEO = "videos/traffic_sign_day_cv.mp4"
TARGET_SECONDS = 6.0  

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1e-6:
    fps = 30.0

target_frame = int(TARGET_SECONDS * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    raise RuntimeError("Could not read the target frame. Try a different TARGET_SECONDS.")

roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
print("ROI (x, y, w, h):", roi)

cv2.destroyAllWindows()
