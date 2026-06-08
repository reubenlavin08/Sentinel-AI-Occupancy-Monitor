# AI Smart Security System — Research & Strategy

> Compiled 2026-06-03. Scope: building a **fully-local, private, multi-camera AI home-security system** that starts as an AI vision + face-recognition baseline and grows toward 3D spatial tracking and behavioral understanding.
> Hardware: Windows 11, Python 3.12, **NVIDIA GTX 1650 (4 GB VRAM)** + Intel UHD 630, 3 IP cameras (2× Android IP-Webcam MJPEG, 1× iOS IP-Camera-Lite H.264/FLV+RTSP). Builder: strong classical CV / 3D-math / embedded, **new to deep learning**.
> Tags used below: **[SHIPPED]** buyable/running today · **[ANNOUNCED]** stated not GA · **[FORECAST]** analyst projection · **[RESEARCH]** lab/paper stage · **[EST]** my estimate · **[HYPE]** marketing > reality.

---

## TL;DR

- **Baseline stack (build first):** YOLO11n (→YOLO26n later) detection + ByteTrack tracking + InsightFace face recognition, all on the GPU; roll-your-own Python pipeline; go2rtc for stream ingest; SQLite events + FFmpeg clips; overlay results on the existing CRT dashboard via WebSocket.
- **The 4 GB GPU is the binding constraint** — use nano/small models, TensorRT FP16, run inference at ~5–8 fps/camera, train from pretrained (not from scratch).
- **The unclaimed opportunity** (no consumer product does it): **cross-camera 3D spatial reasoning + behavioral/anomaly understanding + camera-radar sensor fusion, fully local.** This maps exactly to the builder's rare assets (multiple synced angles, 3D math, embedded/sensor-fusion).
- **The future favors this approach:** regulation (EU AI Act, BIPA, BC PIPA), market trust (cloud-cam breaches), and standards (Matter 1.5, ONVIF) are all pushing toward local-first, private processing. Face recognition is the one feature regulators clamp down on — scope it to the household.

---

## 1. Hardware reality (read first)

- **GTX 1650 = 4 GB VRAM, Turing, 896 CUDA cores.** Real CUDA GPU, but the 4 GB ceiling rules out large models / heavy batching / training big nets.
- Likely **no NVENC encoder** on this SKU (verify) → avoid server-side video re-encoding; prefer sending overlay *metadata* to the browser over re-streaming annotated video.
- **Has NVDEC** (decode) → offload stream decoding to the GPU.
- Practical rules: **nano/small models, TensorRT FP16 (`half=True`), imgsz 416–480, batch the 3 frames, infer every 2nd–3rd frame.** Inference is fine; *training* multi-view nets from scratch is tight (downscale, batch 1, mixed precision) — start from pretrained weights.
- All GTX-1650 FPS figures in this doc are **[EST]** extrapolated from Jetson/other GPUs — benchmark on the real card.

---

## 2. Recommended baseline stack (current best practice, 2026)

### Detection
- **Current landscape:** YOLOv8 (2023) → YOLO11 (Oct 2024) → YOLOv12 (Feb 2025) → **YOLO26** (Jan 14 2026, Ultralytics flagship, native **NMS-free**, ~43% faster CPU on nano). [YOLO26 docs](https://docs.ultralytics.com/models/yolo26)
- **Pick:** start **YOLO11n** (most documented — best while learning; person-detection accuracy ≈ YOLO26n), migrate to **YOLO26n** once the pipeline works (NMS-free = one less post-processing step). **Nano** size for 4 GB + 3 streams.
- **Deploy:** export to **TensorRT engine, FP16**, imgsz 480. [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) · [export](https://docs.ultralytics.com/modes/export)
- **License:** Ultralytics YOLO is **AGPL-3.0** (code *and* weights). Fine for private/home use; only triggers source-disclosure if you *distribute* or expose it as a network service to others. [license](https://www.ultralytics.com/license)
- RT-DETR / RF-DETR (transformer detectors) are more accurate but too VRAM-heavy for 3 streams on 4 GB — skip.

### Tracking (multi-object)
- Built into Ultralytics: **ByteTrack** (fastest) and **BoT-SORT** (default; adds ReID + motion compensation, more robust to occlusion, ~30% slower). [track docs](https://docs.ultralytics.com/modes/track)
- **Pick:** start ByteTrack; switch to BoT-SORT if you get ID switches when people cross. One-line config swap. Use `model.track(frame, persist=True, tracker="bytetrack.yaml")`.
- OC-SORT/Deep OC-SORT/BoostTrack are external (BoxMOT) — only if needed.

### Face recognition
- **Pick: InsightFace `buffalo_l`** (SCRFD detector + ArcFace R50 embedding) via **`onnxruntime-gpu`**. Windows-friendly (no dlib/CMake pain), ONNX-native, auto-uses CUDA. [InsightFace](https://github.com/deepinsight/insightface) · [model guide](https://www.insightface.ai/guides/choose-face-recognition-model-and-evaluate)
  - Avoid `face_recognition`/dlib (painful Windows build). Consider **AdaFace** (export to ONNX) as a v2 upgrade if low-res CCTV faces underperform — it's built for low-quality faces. [AdaFace](https://github.com/mk-minchul/AdaFace)
  - Install **only one** of `onnxruntime` *or* `onnxruntime-gpu`. Use `buffalo_l` (R50), not the heavier `antelopev2` (R100), on 4 GB.
- **Pipeline:** detect+align (SCRFD/5-landmark warp to 112×112) → ArcFace 512-D embedding → L2-normalize → **cosine similarity** vs enrolled embeddings.
- **Enrollment:** 15–30 images/person across angles/lighting/glasses; store all embeddings (or a few centroids) in a local DB.
- **Threshold:** start cosine **~0.35–0.40**; tune on your own validation split, then freeze. Above → that person; below → "unknown."
- **When to run:** NOT every frame. Run on **new confirmed tracks** + opportunistically when a higher-quality face crop appears; gate on min face size/sharpness. **Vote per track** so the name sticks even when the face turns away.

### Architecture (roll-your-own, not a framework)
- **Don't use Frigate** here: not officially Windows-supported (needs fragile WSL2 GPU passthrough) and its face recognition is built-in/locked (can't plug in InsightFace). Roll-your-own Python fits Windows + custom face rec + the learning goal + future 3D/VLM features. [Frigate WSL2 caveat](https://github.com/blakeblackshear/frigate/discussions/4375)
- **Use go2rtc** (single Windows binary) to ingest all 3 feeds — especially to normalize the iPhone FLV/RTSP into one clean RTSP — and to serve clean video to the browser via WebRTC. Your Python reads from go2rtc. [go2rtc](https://go2rtc.org/)
- **Capture:** one thread per camera keeping only the **latest frame** (deque maxlen=1 / one-slot buffer), not a growing queue. For the iPhone use its **RTSP (:8554)**; `ffmpegcv` ([repo](https://github.com/chenxinfeng4/ffmpegcv)) handles RTSP/FLV with no-buffer latest-frame reads.
- **Overlay to dashboard:** annotate **server-side as metadata** — send detection boxes/names/track-IDs as **JSON over WebSocket**; browser draws on a canvas over the video. Avoids GPU re-encoding (no NVENC). FastAPI is a good WebSocket+REST host.
- **Events + clips:** **SQLite** events table (timestamp, camera, class, track-id, name, bbox, clip path, embedding blob) + **FFmpeg `-c copy`** segment recording (no re-encode), record-on-event with pre/post-roll.
- **Cross-camera re-ID:** **OSNet** (via [torchreid](https://github.com/KaiyangZhou/deep-person-reid)) embeddings as a *soft* "likely same person" hint; **face recognition is the reliable identity signal**. Upgrade to CLIP-ReID later if needed. Robust full re-ID across 3 non-overlapping home cams is genuinely hard — keep expectations modest.

### Privacy / legal (BC, Canada — general info, not legal advice)
- BC **PIPA** and federal **PIPEDA** have a **personal/domestic-use exemption** — a homeowner recording their own property for personal security is largely outside the organizational regime. [BC PIPA s.3(2)(a)](https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/00_03063_01)
- **Facial recognition is the line.** It's the single most-regulated feature (BIPA, EU AI Act, the BC Clearview rulings). Scope face-ID to **consenting household members only**, keep it **off any public-facing field of view**, don't capture neighbours, post a surveillance notice, no audio without consent, encrypt the face DB at rest. [Clearview/BC ruling](https://www.torys.com/our-latest-thinking/publications/2025/04/bc-court-requires-clearview-ais-facial-recognition-software-to-comply-with-provincial-privacy-laws)

---

## 3. Cutting-edge research frontier (build toward these)

### Thread A — Multi-view 3D spatial tracking *(most novel; suits 3D-math strength)*
Core idea: warp each camera's CNN features onto a shared floor plane via **homography** → a bird's-eye-view (BEV) occupancy grid; people are detected **once in world coordinates**. Classic geometry + a stock CNN.
- **[BUILD] TrackTacular** — [repo](https://github.com/tteepe/TrackTacular) (2024). Clone first; MVDet/EarlyBird backends, BEV detection→tracking, pretrained weights.
- **[BUILD] MVDeTr** — [repo](https://github.com/hou-yz/MVDeTr) ([arXiv 2108.05888](https://arxiv.org/abs/2108.05888)). Trains on one mid GPU; cleanest to understand.
- **[READ/BUILD] Probabilistic Occupancy Volume** — [arXiv 2503.10982](https://arxiv.org/abs/2503.10982) (CVPR-W 2025). Fuses classical **visual hull** with deep net — most aligned with builder's background. Verify code.
- **[BUILD] GMVD** — [repo](https://github.com/jeetv/GMVD) ([arXiv 2109.12227](https://arxiv.org/abs/2109.12227)). Generalize to a *different* camera rig + indoor synthetic dataset (the real-world hurdle).
- **[BUILD] Self-calibration via people** — [arXiv 2209.07393](https://arxiv.org/abs/2209.07393). People walking calibrate the cameras (matched pose keypoints + factor-graph) — classical geometry + a pose model.
- Datasets: WILDTRACK (real), MultiviewX (synthetic), GMVD (indoor). Use pretrained to validate, then adapt to your rig.
- Feasibility: **inference OK on 4 GB; training tight.** Driving-scale BEV (BEVFormer/BEVFusion) = inspiration only, too heavy.

### Thread B — Open-vocabulary detection *(fastest "wow"; real-time on 1650)*
Detect anything from a **text prompt**, no retraining.
- **[BUILD] YOLOE** — [repo](https://github.com/THU-MIG/yoloe) ([arXiv 2503.07465](https://arxiv.org/abs/2503.07465), ICCV 2025). Text + visual + prompt-free, YOLO-speed, Ultralytics-packaged. Top pick.
- **[BUILD] YOLO-World** — [repo](https://github.com/AILab-CVC/YOLO-World) ([arXiv 2401.17270](https://arxiv.org/abs/2401.17270), CVPR 2024). Predecessor, also real-time.
- Avoid: Grounding DINO 1.5/1.6, DINO-X (API-only, not local). Verify T-Rex2 weights before relying.

### Thread C — VLM behavior understanding + "learn-normal" *(the smartest layer)*
Pattern across all papers: cheap detector flags moments → a **VLM/LLM explains/triages** only those. Run the *pattern* with small local models.
- **[BUILD] Florence-2** — [HF, MIT](https://huggingface.co/microsoft/Florence-2-base). Tiny (~0.7 GB), caption/detect/OCR. Best gentle DL on-ramp; runs great on 1650.
- **[BUILD] AnomalyRuler** — [repo](https://github.com/Yuchen413/AnomalyRuler) (ECCV 2024). LLM writes plain-English "normal" rules from a few-shot of normal frames, flags deviations. *The blueprint for "learn my household's normal."*
- **[REF] LAVAD** ([repo](https://github.com/lucazanella/lavad), CVPR 2024) + **EventVAD** ([arXiv 2504.13092](https://arxiv.org/abs/2504.13092), 2025) — training-free VLM anomaly detection; EventVAD's "segment events before calling the VLM" saves compute.
- **[BUILD] Local small VLMs** for event captioning (run 4-bit via Ollama/llama.cpp): **Moondream2**, **Qwen2.5-VL-3B** (edge of 4 GB), **SmolVLM-500M** (cheap filter).
- **[READ] Holmes-VAU** ([repo](https://github.com/pipixin321/HolmesVAU), CVPR 2025 Highlight) — best explainable VAD, needs a big GPU. Ideas only.
- **Caution [HYPE]:** small VLMs are unreliable as autonomous threat *deciders* ([arXiv 2510.23190](https://arxiv.org/html/2510.23190v1)) — use them as a triage/explanation layer over a deterministic detector, never the detector.

### Enabling pieces (all run on 1650)
- **EdgeTAM** — [repo](https://github.com/facebookresearch/EdgeTAM) ([arXiv 2501.07256](https://arxiv.org/abs/2501.07256), CVPR 2025). Efficient SAM 2: click a person → it follows them. Only "track-anything" model viable on 4 GB (full SAM 2 = offline-only here).
- **Skeleton action recognition** (falls/loitering): **PYSKL/PoseC3D** ([repo](https://github.com/kennymckormick/pyskl)) + ready-made [Human-Falling-Detect-Tracks](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks). Tiny models. *Loitering/tampering: do with classical CV (track-timer / blur-scene-change), not neural nets.*

---

## 4. Future of the field (where the puck is going)

### 4.1 Consumer / industry direction
**Table-stakes now [SHIPPED]:** on-device person/vehicle/package/pet detection (free on non-cloud brands); familiar-faces (Ring cloud Dec 2025 — [TechCrunch](https://techcrunch.com/2025/12/09/amazons-ring-rolls-out-controversial-ai-powered-facial-recognition-feature-to-video-doorbells/); eufy **BionicMind** local; UniFi local); license-plate (prosumer UniFi); mmWave presence as standalone sensors (Aqara FP2/FP300).

**Arriving next (1–3 yr) [SHIPPED→spreading]:** generative-AI **natural-language search + event summaries** moving from cloud-premium → **on-device, no-subscription**:
- Google **Gemini for Home** (Ask Home / Home Brief, Oct 2025, top tier). Ring **Smart Video Search** (premium). **Reolink ReoNeura** — *on-device LLM NL search, no cloud/subscription* (CES Jan 2026, [CNX](https://www.cnx-software.com/2026/01/08/reolink-floodlight-4k-smart-poe-security-cameras-add-on-device-llm-for-natural-language-video-search/)). UniFi **NeXT AI** (local NL search). Verkada **AI Search** (enterprise maturity ceiling).
- **Ring App Store** (Mar 2026) — third-party AI apps, but bans third-party face-rec/LPR.
- Single-device dual-lens auto-handoff (Baseus X1 Pro).

**Still an OPEN GAP (the real opening):** cross-camera identity/spatial tracking across a home; multi-camera **3D** scene fusion; **behavioral/anomaly** understanding ("unusual for a Tuesday 3am"); **camera+radar fusion** in one unit; a **local, cross-camera, NL + behavioral** layer over a self-hosted rig. ([the-gadgeteer 2026 review](https://the-gadgeteer.com/2026/05/11/best-ai-security-cameras-2026/))

**Market [FORECAST]:** AI smart-camera market ~$8–10B (2025) → ~$16–33B by 2030–2035, ~11–13% CAGR (sources diverge; AI is the named growth driver). Broader AI video surveillance ~$6.5B (2024)→~$28.8B (2030), ~30% CAGR ([Grand View](https://www.grandviewresearch.com/industry-analysis/artificial-intelligence-ai-video-surveillance-market-report)).

### 4.2 Emerging paradigms (3–5 yr)
| Paradigm | Status | Note |
|---|---|---|
| **LLM "security copilot" (summarize/search/triage)** | **[SHIPPED] in open source** | **Frigate 0.17** ships GenAI review summaries + semantic NL search with a **fully-local Ollama** backend ([docs](https://docs.frigate.video/configuration/semantic_search/)). **Conntour** ($7M seed, Mar 2026) runs **50 cams on one RTX 4090** ([TechCrunch](https://techcrunch.com/2026/03/26/conntour-raises-7m-from-general-catalyst-yc-to-build-an-ai-search-engine-for-security-video-systems/)). NVIDIA **VSS Blueprint**. Highest-leverage place to aim now. |
| **Reactive → proactive/predictive** | Rules [SHIPPED]; prediction [HYPE] | Honest version = anomaly detection over a learned baseline + auto-deterrence. "Knows a break-in is coming" is hype. ([Verkada 2026](https://www.verkada.com/blog/7-predictions-for-the-physical-security-industry-in-2026/)) |
| **Multimodal sensor fusion (cam+mmWave+audio+PIR)** | [RESEARCH], DIY-underused | Biggest false-alarm-reduction lever; cam+mmWave hit 93.8% acc / 1.7 m error ([IEICE](https://globals.ieice.org/en_transactions/information/10.1587/transinf.2023EDP7106/_f)). Plays to builder's ToF/sensor background. |
| **Edge AI / NPUs** | [SHIPPED] | **Hailo-8/8L on Pi 5** is the DIY accelerator of 2026 (Coral no longer recommended for new builds). Google **Coral NPU** (Oct 2025) is an *architecture*, not a board yet. |
| **Autonomous/robotic** | Outdoor [SHIPPED premium] (Sunflower drones, FAA BVLOS Nov 2025); indoor [HYPE] (Ring Always Home Cam still not GA after 6 yrs) | DIY-transferable piece = **pan-tilt "slew-to-cue" auto-tracking**. |
| **3D / digital twin (Gaussian Splatting)** | [RESEARCH], not in security yet | PerfCam (2025) = cameras + 3DGS + detection → live digital twin localizing objects in 3D. Highest originality, treat as R&D. |

### 4.3 Privacy / regulation / local-first
- **Regulation tightening on biometrics/cloud, not local object detection:** EU AI Act bans untargeted face-DB scraping + restricts live facial recognition ([Art. 5](https://artificialintelligenceact.eu/article/5/), phasing 2025–2027); US **BIPA (Illinois)/Texas/Washington** = consent-before-face-scan with private lawsuits (Ring's Familiar Faces is **blocked** in IL/TX/Portland); BC **Clearview** rulings read privacy exemptions narrowly.
- **Cloud-cam trust is damaged → local-first movement:** NY AG **$450k eufy** settlement (unencrypted streams, Jan 2025); **Wyze** thumbnail breach; Ring backlash. The self-hosted stack (**Frigate + Home Assistant + Scrypted + local cameras**) now ships **free local AI (face/LPR)** that beats paid cloud features.
- **Standards going local/interoperable:** **Matter 1.5** (Nov 2025) adds camera support (WebRTC, local recording) — early, ~12–18 mo from reliable. **ONVIF/RTSP + Frigate** is the robust open foundation today.
- **Geopolitics:** FCC/NDAA bans on Hikvision/Dahua (US) reinforce "**don't trust the camera firmware — control the network + processing yourself**" (VLAN-isolate, block internet egress, all AI local). Good posture regardless of vendor.
- **Net [OPINION, evidence-backed]:** the future strongly favors a **private/local-first** approach. A future-proof system = **cameras as dumb untrusted sensors (VLAN-isolated, no internet) → all AI local (Hailo/OpenVINO/GPU) → no cloud / E2E-encrypted → object-detection default, face-ID only with deliberate consent → vendor-neutral open standards.**

---

## 5. The strategic opportunity

Consumer security is **single-view, reactive, context-blind, cloud-bound**. The defensible, unclaimed territory is the intersection of three things no consumer product combines:

1. **Cross-camera 3D spatial reasoning** (Thread A) — uses the rare multi-angle setup + 3D math.
2. **Behavioral / anomaly understanding + plain-language querying** (Thread C) — the "smart" frontier, runnable locally with small VLMs.
3. **Camera + radar/ToF sensor fusion** — the biggest robustness/false-alarm lever, leaning on the builder's embedded/sensor-fusion background.

…all delivered **fully local and private** — which every regulatory and market trend now rewards. A *private, 3D-aware security brain that understands behavior and answers questions in plain English* is a genuine research-grade differentiator and a strong portfolio/eng-transfer piece.

---

## 6. Recommended roadmap

1. **P1 — Baseline:** YOLO11n detect + ByteTrack on **one** feed (Tab A) on the GPU. (DL on-ramp.)
2. **P2 — Smart alerts:** person-triggered clips (FFmpeg) + self-hosted **ntfy** phone push; SQLite event log.
3. **P3 — Multi-cam + faces:** all 3 feeds + **InsightFace** enrollment/recognition; overlay boxes/names on the CRT dashboard via WebSocket.
4. **P4 — Always-on + secured:** 24/7 on the brain PC; camera **VLAN** (no internet) + **VPN** remote view; disable router UPnP.
5. **Then pick a frontier:** **B (open-vocab, fastest wow)** → **C (VLM learn-normal, smartest)** → **A (multi-view 3D, most novel)**. Consider **sensor fusion** (mmWave/ToF) as a high-value differentiator. **LLM copilot** (Frigate-0.17-style local Ollama summaries/search) is shippable today and the highest-leverage near-term add.

---

## 7. Key references
- **Detection/tracking:** [YOLO26](https://docs.ultralytics.com/models/yolo26) · [tracking](https://docs.ultralytics.com/modes/track) · [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) · [AGPL license](https://www.ultralytics.com/license)
- **Faces:** [InsightFace](https://github.com/deepinsight/insightface) · [model/threshold guide](https://www.insightface.ai/guides/choose-face-recognition-model-and-evaluate) · [AdaFace](https://github.com/mk-minchul/AdaFace)
- **Architecture:** [go2rtc](https://go2rtc.org/) · [ffmpegcv](https://github.com/chenxinfeng4/ffmpegcv) · [BoxMOT](https://github.com/mikel-brostrom/boxmot) · [torchreid/OSNet](https://github.com/KaiyangZhou/deep-person-reid) · [Frigate docs](https://docs.frigate.video/)
- **Multi-view 3D:** [TrackTacular](https://github.com/tteepe/TrackTacular) · [MVDeTr](https://github.com/hou-yz/MVDeTr) · [Prob. Occupancy Volume](https://arxiv.org/abs/2503.10982) · [GMVD](https://github.com/jeetv/GMVD) · [self-calibration](https://arxiv.org/abs/2209.07393)
- **Open-vocab:** [YOLOE](https://github.com/THU-MIG/yoloe) · [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- **VLM/behavior:** [Florence-2](https://huggingface.co/microsoft/Florence-2-base) · [AnomalyRuler](https://github.com/Yuchen413/AnomalyRuler) · [LAVAD](https://github.com/lucazanella/lavad) · [EventVAD](https://arxiv.org/abs/2504.13092) · [Holmes-VAU](https://github.com/pipixin321/HolmesVAU) · [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) · [PYSKL](https://github.com/kennymckormick/pyskl)
- **Future/industry:** [Frigate semantic search](https://docs.frigate.video/configuration/semantic_search/) · [Conntour](https://techcrunch.com/2026/03/26/conntour-raises-7m-from-general-catalyst-yc-to-build-an-ai-search-engine-for-security-video-systems/) · [Ring Familiar Faces](https://techcrunch.com/2025/12/09/amazons-ring-rolls-out-controversial-ai-powered-facial-recognition-feature-to-video-doorbells/) · [Reolink ReoNeura](https://www.cnx-software.com/2026/01/08/reolink-floodlight-4k-smart-poe-security-cameras-add-on-device-llm-for-natural-language-video-search/) · [Matter 1.5 cameras](https://homekitnews.com/2025/11/20/matter-1-5-finally-adds-support-for-smart-cameras-and-more/) · [EU AI Act Art.5](https://artificialintelligenceact.eu/article/5/)

## 8. Sensor fusion — hardware + audio AI *(your differentiator; plays to embedded/ToF strengths)*
Cameras fail in dark/glare/occlusion. Fusing cheap sensors is the biggest false-alarm-reduction lever and leans on your strengths.

**mmWave radar (presence + position, works in the dark):**
- **HLK-LD2450** (~$8–12) — **recommended.** Tracks up to 3 targets with **X/Y position**, UART 256000 baud, native ESPHome (`ld2450`). Best cheap "confirm a person + rough position in the dark." [ESPHome](https://esphome.io/components/sensor/ld2450/)
- HLK-LD2410 (~$3–7) — presence + 1D distance only (incl. *stationary* via micro-motion), no position. Good cheap presence gate. [ESPHome](https://esphome.io/components/sensor/ld2410/)
- TI IWR6843 eval (~$150–300) — real 3D point cloud + people-counting; Phase-3 upgrade, overkill for v1.
- Reality: mmWave is immune to darkness/glare and detects *stillness* (PIR can't), but **don't design around through-wall** (weak, material-dependent). [through-wall](https://linpowave.com/blog/can-mmwave-see-through-walls)

**Audio event detection (glass-break, scream, gunshot, smoke-alarm, dog-bark — works off-camera/in-dark):**
- **YAMNet** (Apache-2.0, ~3.7M params) — start here; 521 AudioSet classes already include the security sounds, runs real-time on CPU, zero training. HA bridge exists ([AudioClassifier-MQTT](https://github.com/c99koder/AudioClassifier-MQTT), [realtime_YAMNET](https://github.com/SangwonSUH/realtime_YAMNET)). [README](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md)
- **CED-base** (Apache-2.0, ~10M, SOTA-efficient) — upgrade for higher accuracy. [repo](https://github.com/RicherMans/CED) · fine-tune on **ESC-50 / UrbanSound8K**.
- AST/PANNs/BEATs = heavier, GPU-preferred, reference only.

**ToF (your VL53L8CX):** keep as the **fast, low-power doorway tripwire / near-field depth gate** (8×8 zones, ≤4 m, autonomous interrupt) — not a room sensor. Also a liveness/anti-spoof depth cue at a door cam.

**Fusion pattern (decision-level cascade — the right altitude for DIY):** cheap sensor **gates** the camera (radar/PIR/ToF detects → camera runs) → **audio classifies** event type in parallel → **camera confirms** identity when there's light → fuse confidences (`radar_person AND camera_person` = high; `radar_person AND camera_dark` = radar fallback; `audio_glassbreak` escalates regardless). The "sensor-masked overlay" (draw radar X/Y + ToF depth on the frame) is a great debug/alert view and a future feature-fusion channel. [radar–camera survey](https://arxiv.org/html/2410.19872v1)

**Starter add-on (~$15–35 new spend):** HLK-LD2450 + an ESP32 (reads radar UART + ToF I²C → MQTT) + a USB mic; laptop runs the camera detector + YAMNet; your fusion node applies the cascade rules. Everything has a Home Assistant path if you want automations free.

## 9. Edge / always-on hardware + the deep-learning learning path

**If you move the brain off the laptop to a 24/7 box** (all prices 2026 estimates):

| Option | Cost | Does YOLO? | Does face rec? | Power | Beginner fit |
|---|---|---|---|---|---|
| **Jetson Orin Nano Super** | ~$249 | Yes (67 TOPS CUDA) | **Yes — same GPU** | 7–25 W | **Best all-in-one + transferable skills (CUDA/PyTorch/TensorRT)** [NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) |
| **Hailo-8L + Pi 5** | ~$150 | Yes (~11 ms YOLOv6n) | **No** (Hailo = detection only; face rec falls to Pi CPU) | 8–12 W | Cheapest low-power *detector appliance* [Frigate HW](https://docs.frigate.video/frigate/hardware/) |
| **Intel N100/N150 mini-PC + OpenVINO** | ~$150–250 | Yes (~one detector) | CPU/iGPU | 15–20 W | Familiar x86; full Frigate |
| **Used small-GPU SFF PC** | ~$150–350 | Yes (RTX 3050: YOLOv9 ~10 ms) | Yes | 150–250 W | Easiest software, power-hungry |

→ For *you* (new to DL, want face rec + to learn): **Jetson Orin Nano Super** is the best learn-once box (YOLO + InsightFace on one CUDA GPU). Hailo+Pi only if you want a pure low-power detector and keep faces on the laptop. **Coral is no longer recommended** (project stale since 2022, driver rot, locked to old small models). [Frigate](https://docs.frigate.video/frigate/hardware/)

**Learn-and-build path (DL beginner who knows classical CV):**
- **PyTorch:** [learnpytorch.io](https://www.learnpytorch.io/) (free book + 26 h video) — best starting point.
- **YOLO detect+track:** [Ultralytics train docs](https://docs.ultralytics.com/modes/train) + the [Roboflow Colab notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb).
- **Faces:** [InsightFace repo](https://github.com/deepinsight/insightface) (`buffalo_l`).
- **Fine-tuning:** only needed for *custom* classes (stock COCO "person" is plenty to start). Always fine-tune from a pretrained checkpoint; on 4 GB use nano/small + batch 4–8, or **train free on Colab/Kaggle T4** and deploy the `.pt` back.
- **Annotation:** start **Roboflow** (AI-assisted, free tier) → **CVAT** (free, self-host) if you outgrow it.
- **Datasets:** COCO (detection), Market-1501 / MSMT17 (re-ID, via [torchreid](https://kaiyangzhou.github.io/deep-person-reid/datasets.html)), LFW/IJB (face eval), UCF-Crime / ShanghaiTech (anomaly), WILDTRACK / MultiviewX (multi-view). Mostly research-only licenses — verify per page.

## 10. 3D tooling + adjacent frontier sensing

**3D / spatial tooling (buildable now):**
- **Calibration:** per-camera intrinsics with **OpenCV ChArUco**, then **COLMAP** (structure-from-motion) to recover the 3 cameras' relative poses in one frame (targetless — ideal for fixed home cams). Kalibr only if you add an IMU. [COLMAP](https://colmap.github.io/tutorial.html)
- **Floor-plan fusion:** `cv2.findHomography` from ≥4 image↔floor-plan correspondences per camera → project each track's foot-point onto a shared top-down map → associate across cameras by proximity → a Kalman filter handles occlusion hand-offs. This is the practical, buildable version of "3D spatial tracking," and it's pure multi-view geometry (your strength). [walkthrough](https://zbigatron.com/mapping-camera-coordinates-to-a-2d-floor-plan/)
- **3D Gaussian Splatting** (photoreal property model as a spatial substrate): **don't train locally** (needs ~6–8 GB; your 1650 is marginal). Use cloud — **Luma AI / Polycam** (free, phone → splat); [nerfstudio + gsplat](https://github.com/nerfstudio-project/gsplat) if you rent a cloud GPU. COLMAP poses tie the splat back to your cameras.

**Frontier sensing, ranked by accessibility:**
1. **WiFi/RF (CSI) sensing — buildable now, ~$10, most novel-per-dollar.** Presence/motion (and experimentally pose/breathing) *through drywall, no camera*, on ESP32. [ESPectre](https://github.com/francescopace/espectre) · [RuView](https://github.com/ruvnet/RuView) · [esp-csi](https://github.com/espressif/esp-csi). Plays directly to your embedded skills. Presence/motion solid; pose/vitals experimental.
2. **Thermal imaging — buildable, moderate cost.** People in total darkness, spoof-resistant. FLIR **Lepton** (~$99–210); the new **Lepton XDS** (~$239, Feb 2026) packs thermal+RGB fusion in one module. [XDS](https://www.cnx-software.com/2026/02/27/lepton-xds-dual-camera-module-combines-160-x-120-thermal-imager-with-5mp-rgb-camera/)
3. **Event cameras (DVS) — now accessible at the low end.** Async per-pixel, >140 dB HDR, near-zero data when static (privacy + bandwidth win). Entry point: **OpenMV GenX320 (~$300)**; research EVKs $1k+. A real paradigm shift to learn. [OpenMV GenX320](https://openmv.io/products/genx320-camera-module)
4. **Gait recognition** — ID by walk when face is hidden. [OpenGait](https://github.com/ShiqiYu/OpenGait) is buildable software; accuracy is good indoors, **poor in the wild** — experiment, not primary ID.
5. **Privacy-preserving CV** — on-device face/body blur before storage (a must-have default); generative anonymization preserves analytics better than blur. Event cameras are inherently privacy-preserving (no RGB texture).

## Confidence flags
- GTX-1650 FPS numbers are **estimates** — benchmark the real card. Verify the card's **NVENC** presence.
- AGPL (YOLO) only matters if you **distribute**; private home use is fine.
- **Facial recognition** is the regulated feature — household-scoped, off public areas.
- Some 2025–2026 papers have **unconfirmed/absent public code** — verify the repo before planning around any.
- VLMs as **autonomous threat-deciders** are not reliable yet — use as triage layer only.
