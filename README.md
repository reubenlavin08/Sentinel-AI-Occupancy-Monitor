# Sentinel — AI Occupancy Monitor

> Real-time room occupancy tracking powered by YOLOv8-Pose and OpenVINO, with a live Streamlit analytics dashboard.

---

## What It Does

Sentinel connects to an IP camera, uses a pose-estimation AI model to detect and track people, and counts entries and exits as people cross a virtual tripwire. All events are logged to a local SQLite database and visualized in a live web dashboard.

- Detects people using **YOLOv8n-Pose**
- Tracks individuals across frames using **BotSort / ByteTrack**
- Counts **IN** and **OUT** crossings via a shoulder-midpoint tripwire
- Logs every event with a timestamp to **SQLite**
- Displays live occupancy, entry/exit totals, and a time-series chart in **Streamlit**
- Runs the model at full speed using **Intel OpenVINO** acceleration

---

## Project Structure

```
Sentinel/
├── occupancy_reid_pose.py        # Main tracking + logging script
├── dashboard.py                  # Streamlit live dashboard
├── compile_openvino.py           # One-time model compilation utility
├── test_camera.py                # Alternate horizontal-tripwire prototype
├── custom_tracker.yaml           # ByteTrack config (extended track buffer)
├── yolov8n-pose.pt               # Base YOLOv8-Pose model weights
├── yolov8n-pose_openvino_model/  # Compiled OpenVINO model (generated)
└── occupancy_log.db              # SQLite event log (generated at runtime)
```

---

## Setup

### 1. Install dependencies

```bash
pip install ultralytics opencv-python streamlit pandas openvino
```

### 2. Configure your camera

Create a `.env` file or edit the URL directly in `occupancy_reid_pose.py`:

```
http://<username>:<password>@<camera-ip>:<port>/video
```

### 3. Compile the model (first time only)

Converts the PyTorch model to OpenVINO format for faster inference:

```bash
python compile_openvino.py
```

### 4. Run the tracker

```bash
python occupancy_reid_pose.py
```

Press **Q** to quit the live window cleanly.

### 5. Launch the dashboard

In a second terminal:

```bash
streamlit run dashboard.py
```

---

## How the Counting Works

A vertical tripwire is drawn at the horizontal center of the frame with an 80px buffer zone. The system tracks the **shoulder midpoint** of each detected person:

- Person starts **left** of the buffer → crosses **right** → counted as **IN**
- Person starts **right** of the buffer → crosses **left** → counted as **OUT**

Using the shoulder midpoint (rather than the bounding box center) makes the count robust to partial occlusions and camera angles.

---

## Dashboard Preview

The Streamlit dashboard polls the database every 2 seconds and displays:

| Metric | Description |
|---|---|
| Current Occupancy | People currently inside |
| Total Daily Entries | Cumulative IN events |
| Total Daily Exits | Cumulative OUT events |
| Occupancy Over Time | Line chart of the full session |
| Raw Event Log | Last 10 crossing events |

---

## Tech Stack

| Component | Library |
|---|---|
| Object detection & pose | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Multi-object tracking | BotSort / ByteTrack |
| Model acceleration | Intel OpenVINO |
| Video capture | OpenCV |
| Data storage | SQLite |
| Dashboard | Streamlit + Pandas |
