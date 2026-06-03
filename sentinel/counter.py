"""Phase 3 — trustworthy counting.

Three fixes over the original (see baseline.md):
  1. shoulder_midpoint() uses keypoint CONFIDENCE (not the brittle `== 0` check).
  2. TrackScorer — Frigate-style confidence voting: a track only "confirms" once
     the median of its score history (padded to >=3) crosses a threshold, so a
     single-frame false positive never counts.
  3. TripwireCounter — a debounce + RESET state machine: resolves origin only
     outside a center buffer, requires K consistent frames to count, then resets
     the track to its new side so the SAME id can be counted again (fixes the
     'counted'-forever and inside-buffer dead-end bugs). Occupancy clamped >= 0.
"""
import statistics
from .config import CONFIG

# COCO-17 pose indices
L_SHOULDER, R_SHOULDER = 5, 6


def shoulder_midpoint(kpts_xy, kpts_conf, conf_thr):
    """Return (x, y) shoulder midpoint, or None if shoulders aren't reliable.
    Falls back to a single visible shoulder; None if neither is trustworthy."""
    if kpts_conf is None:
        lx, ly = kpts_xy[L_SHOULDER]
        rx, ry = kpts_xy[R_SHOULDER]
        if lx == 0 or rx == 0:
            return None
        return ((lx + rx) / 2.0, (ly + ry) / 2.0)

    cl, cr = kpts_conf[L_SHOULDER], kpts_conf[R_SHOULDER]
    if cl >= conf_thr and cr >= conf_thr:
        return ((kpts_xy[L_SHOULDER][0] + kpts_xy[R_SHOULDER][0]) / 2.0,
                (kpts_xy[L_SHOULDER][1] + kpts_xy[R_SHOULDER][1]) / 2.0)
    if cl >= conf_thr:
        return (kpts_xy[L_SHOULDER][0], kpts_xy[L_SHOULDER][1])
    if cr >= conf_thr:
        return (kpts_xy[R_SHOULDER][0], kpts_xy[R_SHOULDER][1])
    return None


class TrackScorer:
    """Confirms a track only after ~N consistent confident detections."""
    def __init__(self, min_score, threshold, min_hist):
        self.min_score = min_score
        self.threshold = threshold
        self.min_hist = min_hist
        self.history = {}
        self.confirmed = set()

    def update(self, tid, score):
        if score < self.min_score:
            return tid in self.confirmed          # ignore weak frame, don't pollute history
        h = self.history.setdefault(tid, [])
        h.append(score)
        if len(h) > 30:
            h.pop(0)
        padded = h + [0.0] * max(0, self.min_hist - len(h))
        if statistics.median(padded) >= self.threshold:
            self.confirmed.add(tid)
        return tid in self.confirmed

    def is_confirmed(self, tid):
        return tid in self.confirmed

    def drop(self, tid):
        self.history.pop(tid, None)
        self.confirmed.discard(tid)


class TripwireCounter:
    """Vertical tripwire at frame center with a +/- buffer dead-zone.
    Per camera; emits 'IN' (crossed to the right) / 'OUT' (crossed to the left)."""
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.line_x = cfg.proc_width // 2
        self.buffer = cfg.buffer_px
        self.k = cfg.cross_confirm_frames
        self.state = {c.id: {} for c in cfg.cameras}      # cam -> {tid: {side, pending, count}}
        self.occupancy = {c.id: 0 for c in cfg.cameras}
        self.entered = {c.id: 0 for c in cfg.cameras}
        self.exited = {c.id: 0 for c in cfg.cameras}

    def update(self, cam, tid, mid_x):
        if mid_x is None:
            return None
        st = self.state[cam]
        s = st.setdefault(tid, {"side": 0, "pending": 0, "count": 0})

        d = mid_x - self.line_x
        side = 0 if abs(d) <= self.buffer else (1 if d > 0 else -1)

        if side == 0:                       # inside dead-zone: wait, don't resolve a side
            s["pending"], s["count"] = 0, 0
            return None
        if s["side"] == 0:                  # first time clearly outside -> adopt origin
            s["side"] = side
            s["pending"], s["count"] = 0, 0
            return None
        if side == s["side"]:               # still on origin side
            s["pending"], s["count"] = 0, 0
            return None

        # crossing candidate (clearly on the opposite side)
        if s["pending"] != side:            # (re)start the K-frame debounce
            s["pending"], s["count"] = side, 1
            return None
        s["count"] += 1
        if s["count"] < self.k:
            return None

        # confirmed crossing
        event = "IN" if side > 0 else "OUT"
        if event == "IN":
            self.entered[cam] += 1
            self.occupancy[cam] += 1
        else:
            self.exited[cam] += 1
            self.occupancy[cam] = max(0, self.occupancy[cam] - 1)   # clamp >= 0
        s["side"] = side                    # RESET: new side is the new origin
        s["pending"], s["count"] = 0, 0
        return event

    def drop(self, cam, tid):
        self.state[cam].pop(tid, None)
