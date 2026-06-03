"""Central configuration. Every magic number that used to be scattered in
occupancy_reid_pose.py lives here. Phase 1 keeps the model on CPU (OpenVINO);
Phase 2 flips `device`/`half`/`model_path` to the GPU PyTorch model.
"""
from dataclasses import dataclass, field
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)


@dataclass
class Camera:
    id: str
    url: str
    fps: int = 15          # used for the per-camera tracker's track_buffer scaling
    rotate: int = 0        # reserved (e.g. iphone 180) — handled upstream by go2rtc for now


@dataclass
class Config:
    # --- cameras: read from the Frigate go2rtc restreams (phone hit once) ---
    cameras: list = field(default_factory=lambda: [
        Camera("tab_a",  "rtsp://localhost:8554/tab_a",   fps=15),   # H.264 restream (go2rtc)
        Camera("tab_e",  "http://192.168.1.76:8080/video", fps=8),   # MJPEG direct (RTSP-wrapping MJPEG is choppy)
        Camera("iphone", "rtsp://localhost:8554/iphone",  fps=10),   # H.264 restream (go2rtc)
    ])

    # --- model (Phase 2 = GPU, PyTorch .pt on the GTX 1650) ---
    model_path: str = os.path.join(REPO_DIR, "yolov8n-pose.pt")  # CUDA needs the torch model, not OpenVINO
    task: str = "pose"
    device: str = "0"          # GTX 1650 (CUDA device 0); "cpu" to fall back
    half: bool = True          # FP16 — real win on the 1650's FP16 cores
    imgsz: int = 640          # 480 gave ~no speedup on the GTX 1650 (overhead-bound), so keep 640 for accuracy
    conf: float = 0.35
    person_class: int = 0

    # --- tracker: finally use custom_tracker.yaml (bytetrack, track_buffer 120) ---
    tracker_yaml: str = os.path.join(REPO_DIR, "custom_tracker.yaml")

    # --- tripwire geometry (faithful to original: vertical center line + buffer) ---
    proc_width: int = 640
    proc_height: int = 480
    buffer_px: int = 40

    # --- motion gating + detect rate (Phase 4) ---
    motion_threshold: int = 25      # pixel-diff threshold on the downscaled gray frame
    motion_min_area: int = 300      # min changed-blob area (downscaled px) to call it motion
    motion_resize: float = 0.25     # downscale factor for the cheap motion diff
    detect_max_fps: float = 6.0     # cap detection rate per camera (plenty for counting)

    # --- counting correctness (Phase 3) ---
    kpt_conf: float = 0.5            # min confidence for a shoulder keypoint to be trusted
    score_min: float = 0.30         # ignore detections below this (don't pollute score history)
    score_threshold: float = 0.50   # median-of-history must exceed this to CONFIRM a track
    score_min_hist: int = 3         # pad score history to >=3 (Frigate-style: ~3 frames to confirm)
    cross_confirm_frames: int = 3   # K consistent frames on the new side to confirm a crossing

    # --- storage ---
    db_path: str = os.path.join(REPO_DIR, "occupancy_log.db")

    # --- event media (Phase 6) ---
    save_events: bool = True
    clip_fps: int = 10              # frames/sec written into clips + buffered
    clip_pre_roll: float = 3.0      # seconds of video kept BEFORE the trigger
    clip_post_roll: float = 3.0     # seconds kept after
    clips_dir: str = os.path.join(REPO_DIR, "clips")
    snapshots_dir: str = os.path.join(REPO_DIR, "snapshots")

    # --- capture ---
    # NOTE: stimeout/timeout naming is FFmpeg-version dependent; kept minimal for
    # reliable opening. The threaded grabber + staleness watchdog handle dead streams.
    rtsp_options: str = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
    stale_timeout: float = 10.0


CONFIG = Config()
