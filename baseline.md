# Baseline — before refactor (2026-06-03)

Captured on the `refactor` branch, machine: i7-9750H (6c/12t), GTX 1650 4GB, 32GB RAM, Win11.

## How it ran before
- **3 separate processes** (`occupancy_reid_pose.py` launched once per camera) reading Frigate restreams `rtsp://localhost:8554/{tab_a,tab_e,iphone}`.
- Each instance: ~570–635 MB RAM (its own full model copy) → **~1.8 GB total**.
- **CPU pegged at 100%** (i7-9750H), because 3 OpenVINO-CPU instances each grabbed all 12 threads and oversubscribed.
- Inference on **CPU only** (torch was `2.12.0+cpu`; the GTX 1650 sat idle).
- FPS per window: low and uneven (single-instance CPU ~8–16 fps; split 3 ways + contention = much lower, choppy).

## Known failures / bugs (to fix in the refactor)
- **Crash on stream blip:** `if not ret: break` exits the whole program on a single failed frame read — all 3 windows died unattended within ~40 min.
- **`custom_tracker.yaml` never used** — code passed `tracker="botsort.yaml"`, so the track_buffer=120 "fix" had no effect.
- **`'counted'` state is terminal** — a track that goes IN then back OUT under the same ID misses the OUT.
- **`spawned_inside_buffer` dead-end** — a person first seen inside the buffer can never be counted.
- **Single-frame trust** — one glitchy frame can cause a false count.
- **`== 0` keypoint check** — brittle; should use keypoint confidence.
- **No occupancy clamp** — can go negative.

## Target after refactor
One process, one shared model on the **GTX 1650** (GPU), per-camera trackers, motion-gated, auto-reconnecting, accurate counting. See `REFACTOR_PLAN.md`.
