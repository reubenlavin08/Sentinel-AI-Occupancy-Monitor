"""Threaded latest-frame RTSP capture with auto-reconnect + staleness watchdog.

Replaces the original `if not ret: break` (which crashed the whole program on a
single dropped frame). Each camera runs a background grabber thread that always
holds only the NEWEST decoded frame; the main loop samples it and never blocks.
"""
import os

# Must be set BEFORE the first cv2.VideoCapture is created — FFmpeg reads it at open time.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)

import cv2
import threading
import time


class CameraStream:
    def __init__(self, url, name, reconnect_delay=2.0, max_delay=30.0, stale_timeout=10.0):
        self.url = url
        self.name = name
        self.reconnect_delay = reconnect_delay
        self.max_delay = max_delay
        self.stale_timeout = stale_timeout

        self._cap = None
        self._frame = None
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort; threaded drain is the real fix
        except Exception:
            pass
        return cap

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"cap-{self.name}", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        delay = self.reconnect_delay
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    self._cap = None
                    time.sleep(delay)
                    delay = min(delay * 2, self.max_delay)
                    continue
                print(f"[{self.name}] stream connected")
                delay = self.reconnect_delay

            try:
                ret, frame = self._cap.read()
            except Exception:
                ret, frame = False, None

            if not ret:
                # never crash — release and reconnect with backoff
                print(f"[{self.name}] read failed; reconnecting...")
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                time.sleep(delay)
                delay = min(delay * 2, self.max_delay)
                continue

            with self._lock:
                self._frame = frame
                self._frame_ts = time.monotonic()

            # staleness watchdog: if frames stop updating, force a reconnect
            if (time.monotonic() - self._frame_ts) > self.stale_timeout:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def read(self):
        """Return (newest_frame, age_seconds). Frame is None if nothing yet."""
        with self._lock:
            if self._frame is None:
                return None, float("inf")
            return self._frame, time.monotonic() - self._frame_ts

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
