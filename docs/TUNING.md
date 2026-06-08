# Frigate tuning notes — this setup (Win11 + Docker Desktop/WSL2, Frigate 0.17)

Findings from a live tuning session (2026-06-03) + research. Sources cited inline.

## The one big lesson: format beats everything

On weak old devices, the **stream format** matters more than any Frigate setting:

- **H.264 (hardware-encoded on the phone) = the winner.** It's cheap for the device (dedicated encoder chip) AND plays *natively* in Frigate's web player (MSE/WebRTC) with **no host transcode**. Smooth, low-latency, high fps.
- **MJPEG = the bottleneck, twice over.** (1) The phone re-compresses every frame as a full JPEG in software — brutal on old CPUs. (2) Browsers can't play MJPEG natively, so Frigate **transcodes it on the host** for live view — this is what triggers the *"Live view is in low-bandwidth mode due to buffering or stream errors"* message.

### What this meant for each camera
| Camera | Best format | Result | Notes |
|--------|-------------|--------|-------|
| **Tab A** (SM-T510) | **H.264 RTSP** `rtsp://IP:8080/h264_ulaw.sdp` | ~15–20 fps, smooth | IP Webcam serves H.264 RTSP by default — no app change needed |
| **iPhone 5S** | **H.264 passthrough** `#video=copy` (IP Camera Lite `/live.flv`) | ~11–20 fps, smooth | Stop re-encoding it; only re-encode if rotation is required |
| **Tab E** (SM-T560NU) | **MJPEG only** `/video` | ~6 fps | Too old (Android 5.1): H.264 endpoint returns **404**. Lever = lower res/quality in app |

### go2rtc source lines that worked
```yaml
go2rtc:
  streams:
    tab_a:  rtsp://192.168.1.144:8080/h264_ulaw.sdp     # H.264, hardware, no transcode
    iphone: ffmpeg:http://192.168.1.136:8081/live.flv#video=copy   # H.264 passthrough
    tab_e:  http://192.168.1.76:8080/video               # MJPEG (only option for this device)
```
Then each camera's `detect` input reads the go2rtc restream `rtsp://127.0.0.1:8554/<name>` with `preset-rtsp-restream` (H.264 cams) or reads MJPEG directly with `preset-http-mjpeg-generic` (Tab E).

## Camera-app settings (the device side)

**IP Webcam (Android):** resolution is the #1 fps lever (MJPEG cost scales with pixels); quality slider is the #2 lever. Use the in-app **orientation/rotate** so you never transcode for rotation. H.264 RTSP endpoints: `/h264_ulaw.sdp` (H.264), `/jpeg_ulaw.sdp` (MJPEG-over-RTSP). H.264 RTSP can be unstable over multi-hour runs — verify it holds.
- **Tab A:** H.264 RTSP, 1280×720, on **5 GHz** (it's Wi-Fi 5 capable).
- **Tab E:** MJPEG, **640×480, quality ~45%**, 2.4 GHz only (Wi-Fi 4) — pick a clear channel (1/6/11), keep it close to the AP.

**IP Camera Lite (iOS, iPhone 5S):** H.264 (NOT HEVC — A7 can't encode HEVC), drop to **640×480**, ~1–2 Mbps, keep app **foreground + screen on + charging** (iOS suspends backgrounded apps / power-saver kills the server). On 5 GHz.

Sources: [Frigate live docs](https://docs.frigate.video/configuration/live/), [camera_setup](https://docs.frigate.video/frigate/camera_setup/), [object_detectors](https://docs.frigate.video/configuration/object_detectors/), [IP Webcam RTSP URLs](https://www.ivideon.com/q2a/16035/support-rtsp-for-the-android-ip-webcam-pro-pavel-khlebovich), [IP Camera Lite App Store](https://apps.apple.com/us/app/ip-camera-lite/id1013455241), [iPhone 5s specs](https://support.apple.com/en-is/111973), [Tab A SM-T510 Wi-Fi 5](https://icecat.biz/en/p/samsung/sm-t510nzkdxsa/), [Tab E SM-T560 Wi-Fi 4](https://icecat.biz/en/p/samsung/sm-t560nzka/).

## Frigate-side knobs (host)

- **`detect.fps: 5`** — recommended. Controls only the *detection pipeline* rate, **NOT** live-view fps (the go2rtc player runs at full stream rate). Higher wastes CPU. (We ran 15 during testing; drop to 5 for efficiency.) [live docs](https://docs.frigate.video/configuration/live/)
- **Detect resolution small** (e.g. 640×360) — the model runs at 300×300 internally; large detect res just wastes CPU.
- **Detector: OpenVINO with `device: CPU`** — faster than the plain `cpu` detector, no GPU passthrough needed. Config:
  ```yaml
  detectors:
    ov: { type: openvino, device: CPU }
  ```
- **OpenVINO GPU / ffmpeg hwaccel: NOT viable here.** Docker Desktop/WSL2 does **not** pass `/dev/dri` (Intel iGPU) into the container reliably; `/dev/dxg` is a DirectX path ffmpeg's VAAPI/QSV can't use. For real GPU accel you'd need bare-metal Linux or a native WSL2 distro with device mapping. [#11133](https://github.com/blakeblackshear/frigate/discussions/11133), [#4375](https://github.com/blakeblackshear/frigate/discussions/4375)
- **shm_size:** 128 MB default is plenty for 3 cams at ≤720p detect. Formula: `(w*h*1.5*20 + 270480)/1048576` MB per cam + ~40 MB logs. [installation](https://docs.frigate.video/frigate/installation/)
- **Motion masks** over the burned-in timestamp bars (both apps stamp a clock) + raise `contour_area` toward 50 to cut wasted detection runs. [masks](https://docs.frigate.video/configuration/masks/), [motion](https://docs.frigate.video/configuration/motion_detection/)
- **Storage on WSL2 ext4, NOT `/mnt/c`** — NTFS-over-9P is dramatically slower for continuous recording and causes clock-skew/mtime warnings. [WSL #4197](https://github.com/microsoft/WSL/issues/4197)

## Running Sentinel alongside Frigate
Frigate is a sealed appliance — you can't run Sentinel's Python *inside* it. Instead, point Sentinel at Frigate's **go2rtc restream** (`rtsp://localhost:8554/tab_a`): the phone is pulled once, both Frigate and Sentinel consume the same hub, Sentinel runs unchanged as your own code. Other hooks: Frigate **MQTT** events and its **HTTP/WS API**. Note Frigate 0.17 has built-in **zone occupancy counting** (MQTT) — Sentinel's custom tripwire logic is the part that's still uniquely yours.
