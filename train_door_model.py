from ultralytics import YOLO

# Define the path to the dataset YAML file
data_yaml_path = 'data/door.yaml'

# Load the YOLOv8 nano model for training
model = YOLO('yolov8n.pt')

# Train the model on the custom door dataset using the CPU
model.train(
    data=data_yaml_path,  # YAML file with dataset configuration
    epochs=50,            # Number of epochs (adjust as needed)
    imgsz=640,            # Image size
    batch=16,             # Batch size
    device='cpu'          # Explicitly use the CPU for training
)

# Save the trained model to a file
model_path = 'trained_door_model.pt'
model.save(model_path)

print(f"Model saved to {model_path}")
