import os
import cv2
import time
from ultralytics import YOLO

# 1. Absolute Pathing
script_dir = os.path.dirname(os.path.abspath(__file__))
# Switched to the Pose model
model_path = os.path.join(script_dir, 'yolov8n-pose.pt') 
tracker_path = os.path.join(script_dir, 'custom_tracker.yaml')

print(f"Loading YOLOv8-Pose AI model from: {model_path} ...")
model = YOLO(model_path) 

# 2. Network Stream Connection
url = "http://<USERNAME>:<PASSWORD>@<CAMERA_IP>:<PORT>/video"
print(f"Attempting to connect to {url}...")

stream = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not stream.isOpened():
    print("ERROR: Could not open the video stream.")
    input("Press Enter to exit...")
    exit()

window_name = "AI Occupancy Monitor (Pose Edition)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

track_state = {} 
entered_count = 0
exited_count = 0
prev_time = 0

while True:
    ret, frame = stream.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (640, 480))
        
    # --- AI INFERENCE (Pose) ---
    results = model.track(source=frame, classes=[0], conf=0.35, persist=True, tracker=tracker_path, imgsz=320, show=False)
    annotated_frame = results[0].plot()
    
    height, width, _ = frame.shape
    
    # --- HORIZONTAL BUFFER ZONE (Adjusted to 50% for 45-degree angle) ---
    tripwire_y = int(height * 0.50) 
    buffer_top = tripwire_y - 40
    buffer_bottom = tripwire_y + 40
    
    cv2.line(annotated_frame, (0, buffer_top), (width, buffer_top), (255, 0, 0), 2)
    cv2.line(annotated_frame, (0, buffer_bottom), (width, buffer_bottom), (255, 0, 0), 2)
    
    # --- KEYPOINT COUNTING LOGIC ---
    # We must check that both boxes AND keypoints exist
    if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
        track_ids = results[0].boxes.id.cpu().numpy()
        # Extract the keypoints array (Shape: [number_of_people, 17, 2])
        keypoints = results[0].keypoints.xy.cpu().numpy() 
        
        for track_id, kp in zip(track_ids, keypoints):
            # kp[0] is the Nose coordinates [x, y]
            nose_x, nose_y = int(kp[0][0]), int(kp[0][1])
            
            # If the nose isn't visible, YOLO returns 0,0. We skip tracking if the nose is lost.
            if nose_x == 0 and nose_y == 0:
                continue

            # Draw a highly visible circle directly on the nose
            cv2.circle(annotated_frame, (nose_x, nose_y), 8, (0, 0, 255), -1)
            
            # State Machine now uses nose_y instead of bounding box cy
            if track_id not in track_state:
                if nose_y < buffer_top:
                    track_state[track_id] = 'originated_top'
                elif nose_y > buffer_bottom:
                    track_state[track_id] = 'originated_bottom'
                else:
                    track_state[track_id] = 'spawned_inside_buffer'

            if track_state[track_id] == 'originated_top' and nose_y > buffer_bottom:
                entered_count += 1
                print(f"Person {track_id} fully crossed DOWN (Entered). Total In: {entered_count}")
                track_state[track_id] = 'counted'
                
            elif track_state[track_id] == 'originated_bottom' and nose_y < buffer_top:
                exited_count += 1
                print(f"Person {track_id} fully crossed UP (Exited). Total Out: {exited_count}")
                track_state[track_id] = 'counted'

    # --- FPS CALCULATION ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    current_occupancy = entered_count - exited_count
    cv2.putText(annotated_frame, f"In: {entered_count}  Out: {exited_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Occupancy: {current_occupancy}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow(window_name, annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()