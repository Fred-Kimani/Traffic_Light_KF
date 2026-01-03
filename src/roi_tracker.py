import cv2
import numpy as np

class RoiTemplateTracker:
    def __init__(self, init_frame: np.ndarray, init_roi: tuple[int,int,int,int]):
        x, y, w, h = init_roi
        self.w, self.h = w, h
        self.template = cv2.cvtColor(init_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        self.last_roi = init_roi

    def update(self, frame: np.ndarray, search_margin: int = 80) -> tuple[int,int,int,int]:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = self.last_roi
        # define a search window around last roi
        sx0 = max(0, x - search_margin)
        sy0 = max(0, y - search_margin)
        sx1 = min(gray.shape[1], x + w + search_margin)
        sy1 = min(gray.shape[0], y + h + search_margin)

        search = gray[sy0:sy1, sx0:sx1]
        if search.shape[0] < h or search.shape[1] < w:
            return self.last_roi

        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # If match is weak, keep previous ROI
        if max_val < 0.72:
         return self.last_roi


        new_x = sx0 + max_loc[0]
        new_y = sy0 + max_loc[1]
        self.last_roi = (new_x, new_y, w, h)
        return self.last_roi
