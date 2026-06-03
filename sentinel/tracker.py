"""One tracker instance PER camera, built the same way Ultralytics builds its
internal per-stream trackers. This is what lets a single shared model serve
multiple cameras without corrupting track IDs across streams — and it finally
loads custom_tracker.yaml (bytetrack, track_buffer=120), which the original
script silently ignored.
"""
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from ultralytics.trackers import BYTETracker, BOTSORT

TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def make_tracker(yaml_path, frame_rate=15):
    data = YAML.load(check_yaml(yaml_path))
    cfg = IterableSimpleNamespace(**data)
    if cfg.tracker_type not in TRACKER_MAP:
        raise ValueError(f"Unsupported tracker_type: {cfg.tracker_type}")
    # ultralytics 8.4.x: BYTETracker(args) reads frame_rate from args (track_buffer
    # is scaled by it). Inject it so track_buffer=120 scales per-camera.
    if not hasattr(cfg, "frame_rate"):
        cfg.frame_rate = frame_rate
    return TRACKER_MAP[cfg.tracker_type](args=cfg)
