"""ONE shared YOLO-Pose model on the GPU (Phase 2).

Runs all cameras' frames through a SINGLE batched predict call — amortizes
launch overhead and is both faster and lower-VRAM than per-camera models.
Tracking is NOT done here (Ultralytics' built-in tracker is single-stream);
each camera owns its own tracker (see tracker.py), fed from these results.
"""
import torch
from ultralytics import YOLO
from .config import CONFIG


class Detector:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        # Resilient device selection: fall back to CPU if CUDA isn't available
        # (e.g. the dGPU dropped after a sleep/resume). Never hard-crash.
        self.device = cfg.device
        self.half = cfg.half
        if str(cfg.device) != "cpu" and not torch.cuda.is_available():
            print("WARNING: CUDA not available — falling back to CPU (slower). "
                  "Reboot / re-enable the NVIDIA adapter to restore GPU.")
            self.device = "cpu"
            self.half = False
        print(f"Loading model from {cfg.model_path} (device={self.device}, half={self.half}) ...")
        self.model = YOLO(cfg.model_path, task=cfg.task)

    def predict_batch(self, frames):
        """frames: list[np.ndarray] (one per camera). Returns list[Results], same order."""
        if not frames:
            return []
        return self.model.predict(
            frames,
            classes=[self.cfg.person_class],
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )

    def predict(self, frame):
        """Single-frame convenience (used by the smoke test)."""
        return self.predict_batch([frame])[0]
