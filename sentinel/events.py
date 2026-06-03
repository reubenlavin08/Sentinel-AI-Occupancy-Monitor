"""Event media (Phase 6, Frigate-style).

ClipBuffer keeps a rolling ring buffer of recent frames per camera, so when a
crossing fires we can write a clip that starts BEFORE the trigger (pre-roll) and
runs a few seconds after (post-roll). BestFrame keeps the single highest-quality
frame of the person across the event (confidence x size, penalized at the edge).
"""
import collections
import cv2


class ClipBuffer:
    def __init__(self, fps=10, pre_roll=3.0, post_roll=3.0):
        self.fps = fps
        self.pre = pre_roll
        self.post = post_roll
        maxlen = int(fps * (pre_roll + post_roll)) + fps  # + headroom
        self.buf = collections.deque(maxlen=maxlen)        # (ts, frame)

    def add(self, ts, frame):
        self.buf.append((ts, frame))

    def save(self, path, trigger_ts, w, h):
        start, end = trigger_ts - self.pre, trigger_ts + self.post
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        n = 0
        for ts, f in list(self.buf):
            if start <= ts <= end:
                vw.write(f)
                n += 1
        vw.release()
        return n


class BestFrame:
    def __init__(self):
        self.score = -1.0
        self.frame = None

    def consider(self, frame, confidence, bbox, w, h):
        x1, y1, x2, y2 = bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        edge = (x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2)
        score = confidence * area * (0.5 if edge else 1.0)
        if score > self.score:
            self.score = score
            self.frame = frame.copy()

    def save(self, path):
        if self.frame is not None:
            cv2.imwrite(path, self.frame)
            return True
        return False

    def reset(self):
        self.score = -1.0
        self.frame = None
