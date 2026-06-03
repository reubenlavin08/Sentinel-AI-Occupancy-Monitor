"""Quick GPU vs throughput benchmark for the batched detector."""
import time
import numpy as np
from sentinel.config import CONFIG
from sentinel.detector import Detector

det = Detector(CONFIG)

# 3 frames (one per camera) at processing resolution
frames = [np.random.randint(0, 255, (CONFIG.proc_height, CONFIG.proc_width, 3), dtype=np.uint8)
          for _ in CONFIG.cameras]

print(f"device={CONFIG.device} half={CONFIG.half} imgsz={CONFIG.imgsz} batch={len(frames)}")

# warm-up (first call compiles/loads kernels)
for _ in range(3):
    det.predict_batch(frames)

N = 30
t0 = time.time()
for _ in range(N):
    det.predict_batch(frames)
dt = time.time() - t0

per_batch_ms = dt / N * 1000
print(f"avg per BATCH (3 frames): {per_batch_ms:.1f} ms  ->  {N/dt:.1f} batches/s")
print(f"effective per-camera: {per_batch_ms/len(frames):.1f} ms/frame  ->  {len(frames)*N/dt:.1f} frames/s total")
