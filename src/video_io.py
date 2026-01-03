import cv2

def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap

def get_fps(cap: cv2.VideoCapture, fps_fallback: float = 30.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 1e-3 else fps_fallback

def get_frame_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h

def make_writer(path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    # MP4 output is easiest to share across OS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer
