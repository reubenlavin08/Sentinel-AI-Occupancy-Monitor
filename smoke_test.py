"""Phase-1 smoke test: validates wiring without opening any GUI windows.
Starts streams, loads the model, runs one predict + tracker.update per camera.
"""
import time
from sentinel.config import CONFIG
from sentinel.capture import CameraStream
from sentinel.detector import Detector
from sentinel.tracker import make_tracker

print("1) building detector + trackers...")
det = Detector(CONFIG)
trackers = {c.id: make_tracker(CONFIG.tracker_yaml, c.fps) for c in CONFIG.cameras}
print("   OK")

print("2) starting streams...")
streams = {c.id: CameraStream(c.url, c.id).start() for c in CONFIG.cameras}
time.sleep(7)  # let them connect

for c in CONFIG.cameras:
    frame, age = streams[c.id].read()
    if frame is None:
        print(f"   [{c.id}] NO FRAME yet (age={age})")
        continue
    import cv2
    frame = cv2.resize(frame, (CONFIG.proc_width, CONFIG.proc_height))
    result = det.predict(frame)
    n = len(result.boxes) if result.boxes is not None else 0
    print(f"   [{c.id}] frame OK (age={age:.1f}s)  persons detected={n}")
    if n:
        d = result.boxes.cpu().numpy()
        tracks = trackers[c.id].update(d, frame)
        print(f"        tracker rows={len(tracks)}", (tracks[0].tolist() if len(tracks) else ""))

for s in streams.values():
    s.stop()
print("DONE")
