# Reliable streaming from old phones over weak outdoor Wi-Fi

Researched 2026-06-07 after repeated drops + "ghost client" pileups with the current
**pull** setup (PC/Frigate pulls HTTP-FLV from each phone). Two root problems and their fixes.

## Root cause of the mess
- **Pull = the PC opens a connection TO the phone.** On weak Wi-Fi a pull drops but the phone's tiny server doesn't release it → "ghost clients" stack up; meanwhile the puller retries and opens more → the phone overloads and serves garbage to everyone. (Each Frigate restart also left a ghost.)
- **HTTP-FLV rides TCP**, which stalls the whole stream on any packet loss (head-of-line blocking) — exactly the freezing seen outdoors.

## THE FIX (architecture): PUSH, not PULL
Have each **phone push ONE outbound stream to a media server on the PC**; Frigate reads from the server, never from the phone.

```
phone (Larix, SRT) ──push──> MediaMTX on PC ──RTSP──> Frigate (go2rtc) ──> detect/record/UI
```

Why this fixes everything:
- **One connection per phone, forever.** All viewers read from MediaMTX, so the phone's load is constant — **ghost-client pileup is impossible** (phone has no inbound server being hammered).
- **SRT (Secure Reliable Transport)** is built for lossy links: UDP + selective retransmission within a latency buffer (recovers ~25% loss at 1s buffer) and **auto-reconnects** on a blip — the two things HTTP-FLV/RTMP fail at.
- **Larix Broadcaster** (free, iOS+Android) adds **adaptive bitrate** — it drops quality when the link sags instead of freezing.

### Components
- **PC: MediaMTX** (free single binary / Docker) — accepts pushed SRT/RTMP, republishes RTSP at `rtsp://localhost:8554/<name>` for Frigate. Ports: SRT 8890/udp, RTMP 1935, RTSP 8554.
- **Phones: Larix Broadcaster** (Softvelum, free) — push SRT to MediaMTX. (IP Camera Lite can also push RTMP/SRT on iOS 13+, but Larix is more robust + has ABR + reliable auto-reconnect, and works on the Android tablets too — IP Webcam is **pull-only**, can't push.)

### Recommended stream settings (this is just security video — bias to survival)
- **SRT push, 480p, ~600 kbps, 10–15 fps, H.264, SRT latency buffer 2000 ms, adaptive bitrate ON.**
- Bump latency to 3000–4000 ms if drops persist on a very weak link.

### Frigate change
Point each camera at MediaMTX instead of the phone:
```yaml
go2rtc:
  streams:
    iphone_front: rtsp://<PC_LAN_IP>:8554/iphone_front
cameras:
  iphone_front:
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8554/iphone_front
          input_args: preset-rtsp-restream
          roles: [detect, record]
```

### Caveats
- **iPhone 5S (iOS 12):** current IP Camera Lite needs iOS 13+, and modern Larix may need newer iOS — the 5S may not be able to PUSH. Keep it as a MediaMTX *pulled* source (`source:` + `sourceOnDemand`) so Frigate still doesn't hit it directly, or retire it.
- **Android tablets (IP Webcam):** pull-only → install **Larix** to push, or front with a MediaMTX `source:` pull (one managed connection instead of Frigate's many).

## THE FIX (physical): get real Wi-Fi outdoors
- **Free now:** put outdoor devices on **2.4 GHz** (better range/penetration; low-bitrate streams don't need 5 GHz), router 2.4 GHz channel to 1/6/11, and device-side: disable Wi-Fi power-saving / "smart network switch" / Wi-Fi Assist, keep app foreground + charging, "stay awake while charging."
- **Real fix (~$120 CAD):** a **wired-backhaul outdoor PoE access point** near the cameras — **TP-Link Omada EAP225-Outdoor** (IP65, ~$90 USD/$120 CAD), fed by **Ethernet** (best) or **MoCA** over existing coax, broadcasting the same SSID in **AP mode** off the Telus gateway (no bridge mode needed; leaves Optik TV untouched). This puts a strong transmitter where the cameras are, instead of repeating an already-weak signal.
- **Last resort:** USB-Ethernet adapter on the device (fully wired, no Wi-Fi), or relocate the camera just inside a window.

## Build order
1. Run **MediaMTX** in Docker on the PC.
2. Install **Larix** on the modern iPhone → push **SRT** (480p/600k/15fps/2000ms/ABR) to MediaMTX → confirm `rtsp://PC:8554/<name>` plays in VLC.
3. Repoint Frigate to read MediaMTX. Repeat per camera (Larix on the tablets; 5S as a pulled source).
4. Wi-Fi: 2.4 GHz + power-save off now; add the outdoor AP when ready.

### Sources
MediaMTX https://github.com/bluenviron/mediamtx · MediaMTX SRT publish https://mediamtx.org/docs/publish/srt-clients · Larix Broadcaster https://softvelum.com/larix/ · SRT RFC https://haivision.github.io/srt-rfc/draft-sharabayko-srt.html · SRT over 4G https://maxsharabayko.github.io/blog/posts/srt-4g-streaming-2022/ · Frigate "server in front of cameras" https://github.com/blakeblackshear/frigate/discussions/1623 · IP Camera Lite https://apps.apple.com/us/app/ip-camera-lite/id1013455241 · EAP225-Outdoor https://www.amazon.com/dp/B07953S2FD · TELUS bridge/AP https://forum.telus.com/t5/Home/Bridge-Mode-Using-Your-Own-Router/ta-p/52181
