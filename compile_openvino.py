import os
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'yolov8n-pose.pt') 

print("Downloading dependencies and compiling YOLOv8-Pose to OpenVINO...")
print("This will take a few minutes. Do not close the window.")

# Load the PyTorch model
model = YOLO(model_path)

# Export the model to OpenVINO format
model.export(format='openvino')

print("\nCompilation complete!")
print("Look in your Sentinel folder for a new directory named: 'yolov8n-pose_openvino_model'")