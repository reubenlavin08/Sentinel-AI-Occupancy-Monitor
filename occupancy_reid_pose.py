import os
import cv2
import time
import sqlite3
from datetime import datetime
from ultralytics import YOLO

# 1. Database Initialization (The Vault)
import sys
CAM_LABEL = sys.argv[1] if len(sys.argv) > 1 else 'cam'
CAM_URL = sys.argv[2] if len(sys.argv) > 2 else "http://<USERNAME>:<PASSWORD>@<CAMERA_IP>:<PORT>/video"
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, f'occupancy_log_{CAM_LABEL}.db')

print(f"Connecting to local database at: {db_path}...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the table if it doesn't already exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS traffic_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        event_type TEXT,
        occupancy INTEGER
    )
''')
conn.commit()

# 2. OpenVINO Pathing Setup
model_path = os.path.join(script_dir, 'yolov8n-pose_openvino_model') 
print(f"Loading Compiled OpenVINO Pose model from: {model_path} ...")
model = YOLO(model_path, task='pose') 

# 3. Network Stream Connection
url = CAM_URL
print(f"Attempting to connect to {url}...")
stream = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not stream.isOpened():
    print("ERROR: Could not open the video stream.")
    input("Press Enter to exit...")
    exit()

window_name = f"Sentinel - {CAM_LABEL}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

track_state = {} 

# To keep the UI matching the DB, we query the last known occupancy on startup
# If the DB is empty, we start at 0.
cursor.execute("SELECT occupancy FROM traffic_events ORDER BY id DESC LIMIT 1")
last_record = cursor.fetchone()
current_occupancy = last_record[0] if last_record else 0

entered_count = 0 
exited_count = 0
prev_time = 0

while True:
    ret, frame = stream.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (640, 480))
        
    # --- AI INFERENCE ---
    results = model.track(source=frame, classes=[0], conf=0.35, persist=True, tracker="botsort.yaml", imgsz=640, show=False)
    annotated_frame = results[0].plot()
    
    height, width, _ = frame.shape
    
    # --- VERTICAL BUFFER ZONE ---
    tripwire_x = int(width / 2) 
    buffer_left = tripwire_x - 40
    buffer_right = tripwire_x + 40
    
    cv2.line(annotated_frame, (buffer_left, 0), (buffer_left, height), (255, 0, 0), 2)
    cv2.line(annotated_frame, (buffer_right, 0), (buffer_right, height), (255, 0, 0), 2)
    
    # --- SHOULDER MIDPOINT LOGIC & DATABASE LOGGING ---
    if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
        track_ids = results[0].boxes.id.cpu().numpy()
        keypoints = results[0].keypoints.xy.cpu().numpy() 
        
        for track_id, kp in zip(track_ids, keypoints):
            ls_x, ls_y = int(kp[5][0]), int(kp[5][1])
            rs_x, rs_y = int(kp[6][0]), int(kp[6][1])
            
            if ls_x == 0 or rs_x == 0:
                continue

            mid_x = int((ls_x + rs_x) / 2)
            mid_y = int((ls_y + rs_y) / 2)

            cv2.circle(annotated_frame, (mid_x, mid_y), 8, (0, 0, 255), -1)
            
            if track_id not in track_state:
                if mid_x < buffer_left:
                    track_state[track_id] = 'originated_left'
                elif mid_x > buffer_right:
                    track_state[track_id] = 'originated_right'
                else:
                    track_state[track_id] = 'spawned_inside_buffer'

            # --- IN EVENT (Crossed Right) ---
            if track_state[track_id] == 'originated_left' and mid_x > buffer_right:
                entered_count += 1
                current_occupancy += 1
                
                # Get the exact time
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Write to database instantly
                cursor.execute("INSERT INTO traffic_events (timestamp, event_type, occupancy) VALUES (?, ?, ?)", 
                               (timestamp, 'IN', current_occupancy))
                conn.commit() # Forces the save to the hard drive
                
                print(f"[{timestamp}] Logged IN. Occupancy: {current_occupancy}")
                track_state[track_id] = 'counted'
                
            # --- OUT EVENT (Crossed Left) ---
            elif track_state[track_id] == 'originated_right' and mid_x < buffer_left:
                exited_count += 1
                current_occupancy -= 1
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                cursor.execute("INSERT INTO traffic_events (timestamp, event_type, occupancy) VALUES (?, ?, ?)", 
                               (timestamp, 'OUT', current_occupancy))
                conn.commit()
                
                print(f"[{timestamp}] Logged OUT. Occupancy: {current_occupancy}")
                track_state[track_id] = 'counted'

    # --- FPS CALCULATION ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # --- CLEAN UI OVERLAY ---
    box_width = 250
    box_start_x = width - box_width - 5
    text_start_x = width - box_width + 5
    
    cv2.rectangle(annotated_frame, (box_start_x, 5), (width - 5, 140), (0, 0, 0), -1)
    
    cv2.putText(annotated_frame, f"Session In: {entered_count} Out: {exited_count}", (text_start_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Occupancy: {current_occupancy}", (text_start_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (text_start_x, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    cv2.imshow(window_name, annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Safely close the database connection to prevent corruption
print("Closing database connection...")
conn.close()
stream.release()
cv2.destroyAllWindows()