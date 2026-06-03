"""Orchestrator: one process, all cameras.

Pipeline per loop: newest frame per live camera -> cheap MOTION GATE -> only
moved cameras (<= detect_max_fps) go through ONE batched GPU inference ->
per-camera tracker -> confidence voting -> keypoint-confidence shoulder-midpoint
-> debounce tripwire counter -> log occupancy + save a pre/post-roll clip and a
best-frame snapshot on each crossing. Press Q to quit.
"""
import os
import time
import cv2

from .config import CONFIG
from .capture import CameraStream
from .detector import Detector
from .tracker import make_tracker
from .motion import MotionGate
from .counter import TripwireCounter, TrackScorer, shoulder_midpoint
from .storage import Storage
from .events import ClipBuffer, BestFrame


def main(cfg=CONFIG):
    detector = Detector(cfg)
    streams = {c.id: CameraStream(c.url, c.id, stale_timeout=cfg.stale_timeout).start()
               for c in cfg.cameras}
    trackers = {c.id: make_tracker(cfg.tracker_yaml, c.fps) for c in cfg.cameras}
    gates = {c.id: MotionGate(cfg.motion_threshold, cfg.motion_min_area, cfg.motion_resize)
             for c in cfg.cameras}
    scorers = {c.id: TrackScorer(cfg.score_min, cfg.score_threshold, cfg.score_min_hist)
               for c in cfg.cameras}
    counter = TripwireCounter(cfg)
    storage = Storage(cfg.db_path)

    # event media (Phase 6)
    if cfg.save_events:
        os.makedirs(cfg.clips_dir, exist_ok=True)
        os.makedirs(cfg.snapshots_dir, exist_ok=True)
    clipbufs = {c.id: ClipBuffer(cfg.clip_fps, cfg.clip_pre_roll, cfg.clip_post_roll) for c in cfg.cameras}
    bestframes = {c.id: BestFrame() for c in cfg.cameras}
    ev_active = {c.id: False for c in cfg.cameras}
    ev_trigger = {c.id: 0.0 for c in cfg.cameras}
    ev_until = {c.id: 0.0 for c in cfg.cameras}
    ev_paths = {c.id: (None, None) for c in cfg.cameras}
    last_clip_add = {c.id: 0.0 for c in cfg.cameras}
    last_infer = {c.id: 0.0 for c in cfg.cameras}

    for c in cfg.cameras:
        cv2.namedWindow(f"Sentinel - {c.id}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Sentinel - {c.id}", cfg.proc_width, cfg.proc_height)

    W, H = cfg.proc_width, cfg.proc_height
    center = W // 2
    bl, br = center - cfg.buffer_px, center + cfg.buffer_px
    min_infer_dt = 1.0 / cfg.detect_max_fps
    min_clip_dt = 1.0 / cfg.clip_fps
    prev_t = time.time()
    loop_n = 0
    last_seen = {}

    print("Sentinel running (Phase 6: clips + snapshots). Press Q in a window to quit.")
    try:
        while True:
            now = time.time()
            display, batch = [], []
            for c in cfg.cameras:
                frame, age = streams[c.id].read()
                if frame is None or age > cfg.stale_timeout:
                    continue
                frame = cv2.resize(frame, (W, H))
                display.append((c, frame))
                # feed the clip ring-buffer at a steady rate (loop runs much faster when idle)
                if now - last_clip_add[c.id] >= min_clip_dt:
                    clipbufs[c.id].add(now, frame)
                    last_clip_add[c.id] = now
                # motion gate + detect-rate cap decide whether to run inference this loop
                if (now - last_infer[c.id]) >= min_infer_dt and gates[c.id](frame):
                    batch.append((c, frame))
                    last_infer[c.id] = now

            if not display:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            results = detector.predict_batch([f for _, f in batch]) if batch else []
            infer = {c.id: r for (c, _), r in zip(batch, results)}

            fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
            prev_t = now
            loop_n += 1
            if loop_n % 30 == 0:
                print(f"[perf] loop fps={fps:.1f}  shown={len(display)} inferred={len(batch)}")

            for c, frame in display:
                annotated = frame.copy()
                cv2.line(annotated, (bl, 0), (bl, H), (255, 0, 0), 2)
                cv2.line(annotated, (br, 0), (br, H), (255, 0, 0), 2)

                result = infer.get(c.id)
                if result is not None:
                    det = result.boxes.cpu().numpy() if result.boxes is not None else None
                    kxy = (result.keypoints.xy.cpu().numpy()
                           if result.keypoints is not None else None)
                    kconf = (result.keypoints.conf.cpu().numpy()
                             if (result.keypoints is not None and result.keypoints.conf is not None)
                             else None)

                    if det is not None and len(det) and kxy is not None:
                        tracks = trackers[c.id].update(det, frame)
                        for row in tracks:
                            track_id = int(row[4]); det_idx = int(row[-1]); score = float(row[5])
                            last_seen[(c.id, track_id)] = loop_n
                            x1, y1, x2, y2 = (int(v) for v in row[:4])
                            confirmed = scorers[c.id].update(track_id, score)
                            box_color = (0, 255, 0) if confirmed else (140, 140, 140)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(annotated, f"ID {track_id}", (x1, max(0, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                            if confirmed:
                                bestframes[c.id].consider(frame, score, (x1, y1, x2, y2), W, H)

                            if det_idx >= len(kxy):
                                continue
                            mid = shoulder_midpoint(kxy[det_idx],
                                                    kconf[det_idx] if kconf is not None else None,
                                                    cfg.kpt_conf)
                            if mid is None:
                                continue
                            mid_x, mid_y = int(mid[0]), int(mid[1])
                            cv2.circle(annotated, (mid_x, mid_y), 8,
                                       (0, 0, 255) if confirmed else (140, 140, 140), -1)
                            if not confirmed:
                                continue
                            event = counter.update(c.id, track_id, mid_x)
                            if event:
                                clip_p = snap_p = None
                                if cfg.save_events and not ev_active[c.id]:
                                    ev_active[c.id] = True
                                    ev_trigger[c.id] = now
                                    ev_until[c.id] = now + cfg.clip_post_roll
                                    stamp = int(now)
                                    clip_p = os.path.join(cfg.clips_dir, f"{c.id}_{stamp}.mp4")
                                    snap_p = os.path.join(cfg.snapshots_dir, f"{c.id}_{stamp}.jpg")
                                    ev_paths[c.id] = (clip_p, snap_p)
                                elif cfg.save_events:
                                    ev_until[c.id] = now + cfg.clip_post_roll  # extend window
                                ts = storage.log(c.id, event, counter.occupancy[c.id], clip_p, snap_p)
                                print(f"[{ts}] {c.id}: {event}. Occupancy: {counter.occupancy[c.id]}")

                # flush a finished event: write clip + best-frame snapshot
                if ev_active[c.id] and now > ev_until[c.id]:
                    clip_p, snap_p = ev_paths[c.id]
                    nframes = clipbufs[c.id].save(clip_p, ev_trigger[c.id], W, H)
                    saved = bestframes[c.id].save(snap_p)
                    bestframes[c.id].reset()
                    ev_active[c.id] = False
                    print(f"   saved clip {os.path.basename(clip_p)} ({nframes} frames), snapshot={saved}")

                cv2.rectangle(annotated, (W - 255, 5), (W - 5, 110), (0, 0, 0), -1)
                cv2.putText(annotated, f"{c.id}  In:{counter.entered[c.id]} Out:{counter.exited[c.id]}",
                            (W - 248, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated, f"Occupancy: {counter.occupancy[c.id]}",
                            (W - 248, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                idle = "" if c.id in infer else " (idle)"
                cv2.putText(annotated, f"fps:{int(fps)}{idle}",
                            (W - 248, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(f"Sentinel - {c.id}", annotated)

            if loop_n % 300 == 0:
                for (cam, tid), seen in list(last_seen.items()):
                    if loop_n - seen > 300:
                        scorers[cam].drop(tid); counter.drop(cam, tid); del last_seen[(cam, tid)]

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("Shutting down...")
        for s in streams.values():
            s.stop()
        storage.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
