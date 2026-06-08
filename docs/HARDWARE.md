# Camera hardware plan (research-backed, 2026-06-07)

Conclusion after a long night fighting phone cameras: **repurposed phones are inherently
unreliable as 24/7 cameras** — fine as a free proof-of-concept, not as a foundation.

## Why phones are the problem (not config)
- Phones aren't built to run 24/7 as cameras; dedicated cams give night vision, weather
  resistance, robust operation ([How-To Geek](https://www.howtogeek.com/dont-trade-in-your-old-phone-its-much-better-as-a-dedicated-security-camera/)).
- The exact failures hit: IP "changes all the time", reliability depends on the phone's
  Wi-Fi + the OS suspending apps ([IP Cam Talk](https://ipcamtalk.com/threads/is-there-a-android-turn-your-old-phone-into-a-ip-camera-app-for-nvr-recording.14159/)).
- No IR night vision — phones are half-blind for security ([Reviews.com](https://www.reviews.com/home/security-systems/old-phone-might-be-new-security-camera/)).
- Observed: Tab A's app froze (HTTP+RTSP dead); 5S gets suspended by iOS and cuts in/out;
  newer iPhone (iphone2) stays solid → proves it's the device, not the setup.

## What to buy
Frigate gold standard = **PoE cameras** (Power over Ethernet: one cable = power + data +
network, no Wi-Fi, no power brick). Frigate docs recommend Dahua/Hikvision, then Amcrest;
Reolink also works ([Frigate hardware](https://docs.frigate.video/frigate/hardware/)).

| Camera | ~Price CAD (est) | Notes |
|---|---|---|
| Reolink RLC-510A | ~$95 | 5MP, PoE, RTSP, SD slot — cheap entry |
| Reolink RLC-810A | ~$120 | 4MP, F1.6 lens = better night vision |
| Amcrest IP5M | ~$135 | Officially Frigate-tested, best ONVIF compatibility |

Mix brands freely — Frigate treats every RTSP source the same.
Want H.264 sub-stream for detect (1080p ~2 Mbps) + main stream for record.

## Solving "can't run Ethernet across the property"
- **Point-to-point wireless bridge** — two 5GHz radios making a dedicated wired-equivalent
  link house↔far corner; put a cheap PoE switch at the far end feeding several cams.
  This is the "semi-wireless" setup: wired at the camera, one solid wireless hop back.
  TP-Link pairs ~$110; Ubiquiti = quality pick. **Needs clear line-of-sight.**
  ([FastCabling](https://www.fastcabling.com/2025/01/24/4-ways-to-install-security-cameras-with-outdoor-wireless-bridges/), [GNS Wireless](https://www.gnswireless.com/blog/wireless-bridge-for-security-cameras/))
- **Per-camera PoE-to-Wi-Fi bridge** (Alarm.com ADC-W110, or travel router in client mode)
  — one wired cam rides existing Wi-Fi; cheaper, inherits Wi-Fi flakiness.
- **Outdoor PoE AP** (TP-Link EAP225-Outdoor ~$120) — strong Wi-Fi *near* the cams (see STREAMING.md).

## The plan
1. **Where Ethernet reaches:** 2–3 PoE cams → ~$35 PoE switch → PC. Bulletproof; replaces the tablets.
2. **Far points:** one point-to-point bridge → small PoE switch → cams there.
3. **Detection compute:** the GTX 1650 already handles Frigate detection — **no Coral TPU needed.**
4. **Phones:** keep only as bonus angles on Larix push (self-healing); never load-bearing.
5. **ESP32-S3 cams:** good learning project, not the backbone; build **wired** (PoE/W5500).

**Starter budget:** ~2 PoE cams + switch + cabling ≈ **$250–300 CAD** for a real,
night-vision-capable, never-drops core. Add the bridge (~$110) for the far side later.
