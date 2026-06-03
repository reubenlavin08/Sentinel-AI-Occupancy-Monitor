"""Cheap motion gate (Frigate-style). Runs BEFORE the neural net so idle
cameras skip the expensive inference entirely.

Diffs each frame against a running-average background (not just the previous
frame), so slow lighting drift doesn't register as motion. The background is
only updated when the scene is still, so a person standing in view doesn't get
'learned' into the background.
"""
import cv2


class MotionGate:
    def __init__(self, threshold=25, min_area=300, resize=0.25, frame_alpha=0.05,
                 blur_ksize=(5, 5)):
        self.threshold = threshold        # pixel-diff threshold on the downscaled gray image
        self.min_area = min_area          # min changed-blob area (downscaled px) to call it motion
        self.resize = resize              # downscale factor for the cheap diff
        self.frame_alpha = frame_alpha    # background learning rate (when still)
        self.blur_ksize = blur_ksize
        self.bg = None

    def __call__(self, frame):
        small = cv2.resize(frame, None, fx=self.resize, fy=self.resize,
                           interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_ksize, 0)

        if self.bg is None:
            self.bg = gray.astype("float32")
            return False

        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg))
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moved = any(cv2.contourArea(c) >= self.min_area for c in cnts)

        if not moved:   # only fold the frame into the background when the scene is still
            cv2.accumulateWeighted(gray, self.bg, self.frame_alpha)
        return moved
