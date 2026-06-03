# Sentinel — Refactor & Upgrade Plan (in depth)

**Goal:** turn the single-file, single-camera, CPU, crash-on-blip prototype into a **single-process, GPU-accelerated, multi-camera occupancy counter** that is robust, accurate, and reads from the Frigate go2rtc restreams — borrowing proven techniques from Frigate (MIT-licensed).

**Machine:** Win11, i7-9750H (6c/12t), **NVIDIA GTX 1650 4 GB** (Turing, CC 7.5, FP16 cores, no tensor cores), Intel UHD 630 iGPU. Python 3.12 venv. Frigate runs in Docker/WSL2 and exposes `rtsp://localhost:8554/{tab_a,tab_e,iphone}`.

## Guiding principles
1. **Prune & restructure the existing code FIRST, behavior-preserving — then add features.** (Per your call.)
2. Every phase ends **working, runnable, and measurable**; we review before moving on.
3. **One shared model**, one process. GPU-first for the big perf win.
4. Keep the **original files untouched**; new code lives in a `sentinel/` package on a branch, so the 3-instance demo stays available for comparison.
5. Feed from **Frigate's restreams** (phone hit once; both Frigate and Sentinel consume the same hub).

## Phase → Frigate-technique map (the 6 you saved)
| Frigate technique | Lands in |
|---|---|
| 1. Motion-gated detection | Phase 4 |
| 2. Latest-frame capture (drop stale) | Phase 1 |
| 3. Confidence voting (object scoring) | Phase 3 |
| 4. Crossing debounce / inertia | Phase 3 |
| 5. ~5 fps detect + region crop | Phase 4 |
| 6. Best-frame snapshot + pre/post-roll clips | Phase 6 |

---

## Target architecture
```
sentinel/
  __init__.py
  config.py      # Config dataclass + CAMERAS list (restreams) + all thresholds/geometry
  capture.py     # CameraStream: threaded latest-frame grabber, reconnect, staleness watchdog
  detector.py    # Detector: ONE YOLO-Pose model on the GPU; batched predict over cameras
  tracker.py     # one ByteTracker instance PER camera (loads custom_tracker.yaml); det->kpt mapping
  motion.py      # MotionGate: cheap frame-diff gate (Frigate-style running background)
  counter.py     # TripwireCounter (debounce+reset FSM) + TrackScorer (confidence voting) + keypoint midpoint
  storage.py     # Storage: WAL SQLite, single writer thread fed by a queue; per-camera schema
  events.py      # ClipBuffer (pre/post-roll ring buffer) + BestFrame (highest-score snapshot)
  app.py         # orchestrator: wires capture->motion->detect->track->score->count->store + overlays
dashboard.py     # (existing) adapted to the multi-camera schema
```
**Per-frame flow (per camera):** `CameraStream.read()` → `MotionGate` → (if motion) batched `Detector.predict` → per-camera `tracker.update` → `TrackScorer` (need ~3 confident frames) → shoulder-midpoint (keypoint-confidence gated) → `TripwireCounter` → `Storage` + `events`.

---

## Phase 0 — Safety net & baseline
**Objective:** nothing breaks; we can compare before/after.
- 0.1 `git checkout -b refactor` (local branch; **do not push** without asking).
- 0.2 `pip freeze > requirements-baseline.txt` to capture the working CPU env.
- 0.3 Record the **baseline**: run the current 3-instance demo, note per-window FPS, total RAM (~1.8 GB), CPU (100%), and the crash-on-blip behavior. Save to `baseline.md`.
- 0.4 Confirm original files (`occupancy_reid_pose.py`, `dashboard.py`, models) stay in place; all new work goes in `sentinel/`.
- **Acceptance:** branch exists, baseline documented, original demo still launchable.

---

## Phase 1 — Prune & restructure (behavior-preserving)
**Objective:** same detection/counting *logic*, but clean modular structure, config-driven, one process over all 3 cameras, and it no longer dies on a stream blip. **No new accuracy features yet** (those are Phase 3) — this is the "clean it before adding" phase.

Steps:
- 1.1 Scaffold the `sentinel/` package (empty modules + `__init__.py`).
- 1.2 **`config.py`** — a `@dataclass Config` holding every magic number currently scattered in the script: `imgsz=640`, `conf=0.35`, `buffer_px=40`, `device`, `db_path`, tripwire geometry, plus a `CAMERAS` list:
  ```python
  CAMERAS = [
    Camera("tab_a",  "rtsp://localhost:8554/tab_a",  fps=15),
    Camera("tab_e",  "rtsp://localhost:8554/tab_e",  fps=8),
    Camera("iphone", "rtsp://localhost:8554/iphone", fps=10),
  ]
  ```
  Removes the hardcoded `<USERNAME>...` URL from logic.
- 1.3 **`capture.py`** — the `CameraStream` class (threaded latest-frame + reconnect-with-backoff + staleness watchdog). This is *plumbing*, not a feature, and is the backbone of multi-camera. Set the FFMPEG low-latency env (`OPENCV_FFMPEG_CAPTURE_OPTIONS` with `rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000` + a socket timeout) **before** any capture is created. `read()` returns `(frame, age)`; the loop skips stale/dead cams instead of `break`-ing. *(Fixes the crash that killed your windows.)*
- 1.4 **`detector.py`** — load the model **once**; expose `infer(frames)`. Phase 1 keeps it on **CPU** (OpenVINO) to stay behavior-preserving; GPU is Phase 2.
- 1.5 **`counter.py` (faithful port)** — move the *existing* shoulder-midpoint state machine in verbatim, but keyed per-camera (dict of state per camera). Behavior identical to the original for now, so we can diff. Flag (don't yet fix) the known bugs in comments: `'counted'`-forever lock, `spawned_inside_buffer` dead-end, `==0` keypoint check, no occupancy clamp.
- 1.6 **`storage.py` (faithful port)** — same SQLite writes, one module. (Hardening is Phase 5.)
- 1.7 **`app.py`** — orchestrate: start 3 `CameraStream`s, loop reading latest frames, run detect→count→store, draw the existing overlay, one window per camera (or a combined grid). `python -m sentinel` is the entry point.
- 1.8 **Prune dead code:** remove the unused `custom_tracker.yaml` reference path (we re-introduce it *properly* in Phase 2), dedup `entered_count`/`current_occupancy` bookkeeping, drop the `input("Press Enter")` hard-exit, delete commented cruft. Keep `test_camera.py` as a reference but exclude from the package.
- **Acceptance:** `python -m sentinel` runs **all 3 cameras in ONE process on CPU**, survives a camera dropping/reconnecting, and produces the *same* counts the original did on a test walk-through. Clean modules, everything config-driven.

---

## Phase 2 — GPU acceleration + shared-model multi-camera detection/tracking
**Objective:** the big perf unlock. Move inference to the GTX 1650, share one model across cameras via batched `predict`, and give each camera its own tracker (required once we leave `model.track`).

Steps:
- 2.1 **Install CUDA torch** (replaces the `+cpu` build):
  ```powershell
  pip uninstall -y torch torchvision
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  (No separate CUDA toolkit needed; wheels bundle the runtime. Driver 576.88 ≥ 570.65 floor. Verify exact torch version with `pip index versions torch`.)
- 2.2 **`detector.py` → GPU:** load the **PyTorch** model (`yolov8n-pose.pt`, *not* the OpenVINO dir — CUDA needs the torch model). Run **batched**:
  ```python
  results = model.predict(source=[f0, f1, f2], device=0, half=True,
                          imgsz=cfg.imgsz, classes=[0], verbose=False)
  # results[i] corresponds to camera i, in order
  ```
  Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the long-running process.
- 2.3 **`tracker.py` — one tracker per camera** (since batched `predict` has no built-in tracker):
  ```python
  from ultralytics.utils import YAML, IterableSimpleNamespace
  from ultralytics.utils.checks import check_yaml
  from ultralytics.trackers import BYTETracker, BOTSORT
  TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
  def make_tracker(yaml_name, frame_rate):
      cfg = IterableSimpleNamespace(**YAML.load(check_yaml(yaml_name)))
      return TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=frame_rate)
  trackers = {c.id: make_tracker("bytetrack.yaml", c.fps) for c in CAMERAS}
  # per camera: tracks = trackers[cid].update(results[i].boxes.cpu().numpy(), frame)
  ```
  Use **ByteTrack** (faster, no ReID — right for fixed indoor tripwire cams). This is where your **`custom_tracker.yaml`** (track_buffer 120) finally gets loaded — per camera, with `frame_rate` set so the buffer scales correctly.
- 2.4 **det→keypoint mapping:** the tracker output row's last column is the detection index; map back with `results[i].keypoints[int(row[-1])]`. **Verify the row layout once** by printing a row (`[x1,y1,x2,y2,id,score,cls,det_idx]` — has shifted across versions).
- 2.5 **GPU/Frigate coexistence:** Frigate is on CPU today, so no contention. If Frigate ever moves to GPU, keep it on the **iGPU/OpenVINO** and reserve the GTX 1650 for Sentinel (avoids the 4 GB squeeze).
- 2.6 Measure FPS vs the Phase 0 baseline; expect ~5–10×.
- **Acceptance:** GPU inference confirmed, large FPS jump, independent track IDs per camera (no cross-stream bleed), VRAM < ~2.5 GB (check `nvidia-smi`).

---

## Phase 3 — Counting correctness (the accuracy fixes)
**Objective:** fix the counting bugs and make counts trustworthy. (Frigate techniques #3 + #4.)

Steps:
- 3.1 **Keypoint-confidence midpoint** — replace the `==0` shoulder check with `results.keypoints.conf`; require both shoulders ≥ `KPT_CONF` (fallback to one visible shoulder), else return `None` (skip this frame for this track). COCO indices 5=L, 6=R.
- 3.2 **`TrackScorer` (confidence voting)** — Frigate-style: keep per-track score history, ignore detections below `min_score`, and only "confirm" a track once the **median of its score history (padded to ≥3)** crosses `threshold` (≈3 consistent confident frames). Only confirmed tracks may count. Kills one-frame false positives.
- 3.3 **`TripwireCounter` FSM (debounce + reset)** — signed-distance-to-line state machine that:
  - resolves an origin side **only when clearly outside** the center buffer (no spurious origins from jitter) — *fixes `spawned_inside_buffer`*;
  - requires **K consecutive frames** on the new side before emitting (debounce) — *fixes double-counts*;
  - after a confirmed crossing, **resets the track to its new side** so the same ID can cross again — *fixes the `'counted'`-forever bug*.
- 3.4 **Occupancy clamp** — `max(0, occupancy)`; add a daily/manual reset.
- 3.5 Gate the FSM on `TrackScorer.is_confirmed()`; `.drop(tid)` from scorer/counter when a track expires (bound memory).
- **Acceptance:** walk-through tests — in→out→in on one ID all count; loitering at the line counts once; a flicker false-positive never counts; occupancy never goes negative.

---

## Phase 4 — Motion-gating + detect-rate efficiency (Frigate #1, #5)
**Objective:** stop burning GPU/CPU on empty rooms.
- 4.1 **`MotionGate`** — Frigate-style running-background diff: grayscale → downscale → blur → `absdiff` vs `accumulateWeighted` background → threshold → dilate → `contourArea` filter. Returns `(moved, regions)`. Start `threshold≈25`, `contour_area≈300–500` on a ¼-scale frame; tune.
- 4.2 Gate the detector per camera: skip `predict` when no motion, **but still tick `tracker.update(empty_det, frame)`** so `track_buffer` aging stays correct (verify behavior).
- 4.3 **Detect-rate cap** — sample each camera at ~5 fps for detection regardless of stream fps (decouple from capture/live rate).
- 4.4 Optional **motion masks** — bitwise-AND a per-camera mask to ignore timestamp overlays / TVs / windows.
- 4.5 Optional **region crop** — feed the motion region at model resolution for better small/distant detection.
- **Acceptance:** idle cameras drop GPU/CPU usage to near-zero; detection still fires promptly on entry.

---

## Phase 5 — Storage hardening + multi-camera dashboard
**Objective:** no "database is locked", per-camera analytics.
- 5.1 **`storage.py`** — WAL mode (`journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`), a **single writer thread** fed by a `queue.Queue` (every write serialized; one connection owned by that thread). Schema gains a `camera` column (+ optional `clip_path`, `snapshot_path`).
- 5.2 **`dashboard.py`** — per-camera occupancy + a combined total; reader connection also WAL + busy_timeout; group charts by camera.
- **Acceptance:** sustained writes from 3 cameras + dashboard polling with zero lock errors; dashboard shows per-camera + total.

---

## Phase 6 — Events: pre/post-roll clips + best-frame snapshot (Frigate #6)
**Objective:** save the clearest evidence per event, with lead-in.
- 6.1 **`ClipBuffer`** — per-camera `collections.deque(maxlen=fps*(pre+post))` of `(ts, frame)`; on event, flush from `trigger_ts - pre_roll` through `post_roll` to an mp4. (Watch RAM: ~300 MB at 3×10fps×10s×1MB — store JPEG bytes if tight.)
- 6.2 **`BestFrame`** — keep the highest `confidence × area × edge-penalty` frame across the event (Frigate's scoring), save as the snapshot; add a `best_image_timeout` so long events don't thrash.
- 6.3 Wire crossing events → save clip + snapshot, log paths in `storage`.
- **Acceptance:** each crossing yields a clip starting a few seconds *before* the trigger + a sharp snapshot, both referenced in the DB.

---

## Phase 7 — Optional / future
- 7.1 **TensorRT FP16 export** (`format=engine, half=True`, fixed `imgsz`) for ~1.3–1.6× more — engine is device/version-specific, rebuild on-device, don't commit it.
- 7.2 **Zones** (beyond a single tripwire) and per-camera tripwire geometry.
- 7.3 **Frigate integration** — optionally subscribe to Frigate MQTT events, or publish Sentinel occupancy to MQTT.
- 7.4 The 4 hardening techniques become a reusable base for adding cameras (incl. future ESP32-S3 / ESP32-P4 cams).

---

## Risks & things to verify on-device
- **Exact torch version** on the cu128 channel (`pip index versions torch`); the `2.12.0` seen may be custom.
- **Tracker row layout** `[...,det_idx]` — print one row to confirm before relying on the mapping.
- **FFMPEG option names** (`stimeout` vs `timeout`, `max_delay` units) vary by build — `cap.read()` can block on a half-dead socket, so the socket timeout is the key resilience knob; verify with `ffmpeg -h demuxer=rtsp`.
- **Ticking trackers on motion-gated frames** (Phase 4.2) is an inference about keeping `track_buffer` aging correct — validate.
- **GPU benchmark** is interpolated; run `yolo benchmark model=yolov8n-pose.pt imgsz=640 device=0 half=True` for real numbers.
- **VRAM headroom** vs Frigate — monitor `nvidia-smi`; keep desktop on the iGPU.

## Suggested build order (review gates)
**0 → 1 → 2 → 3** delivers the core product (clean, GPU-fast, robust, accurate multi-cam counter). **4 → 5 → 6** harden and enrich. **7** is later. Each phase is independently runnable and reviewed before the next.
