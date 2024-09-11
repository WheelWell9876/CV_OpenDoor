from flask import Flask, render_template, Response, jsonify
import subprocess
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Create directories for saving images if they don't exist
os.makedirs("data/door_open", exist_ok=True)
os.makedirs("data/door_closed", exist_ok=True)

# Initialize models and camera
camera = cv2.VideoCapture(0)

# Define the path to the trained door detection model
trained_model_path = 'trained_door_model.pt'

# Try loading the trained door detection model if it exists, otherwise load fallback model
door_model = None
if os.path.exists(trained_model_path) and os.path.getsize(trained_model_path) > 0:
    try:
        door_model = YOLO(trained_model_path)  # Load the trained model
        print("Trained door detection model loaded.")
    except Exception as e:
        print(f"Error loading the trained model: {e}. Falling back to YOLOv8 nano model.")
        door_model = YOLO('yolov8n.pt')  # Fallback to default YOLOv8 model
else:
    door_model = YOLO('yolov8n.pt')  # Fallback to YOLOv8 nano model
    print("Trained door model not found. Loaded YOLOv8 nano model as fallback.")

# General YOLOv8 model (default for regular detection)
yolo_model = YOLO('yolov8n.pt')

# Control variables for capture
capture = False
door_state = None
open_counter = 0
closed_counter = 0

@app.route('/')
def index():
    """Render the home page with start/stop buttons and mode switch."""
    return render_template('index.html')

@app.route('/start_capture/<state>', methods=['POST'])
def start_capture(state):
    """Start capturing images for door state (open/closed)."""
    global capture, door_state
    capture = True
    door_state = state
    return jsonify({'status': f'Started capturing for {state} door'})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    """Stop capturing images."""
    global capture
    capture = False
    return jsonify({'status': 'Stopped capturing'})

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start training the model and print the logs to the terminal."""
    print("Training the model...")

    # Trigger the training process and print output to the terminal
    command = ['python', 'train_door_model.py']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Print training logs directly to the terminal
    for line in process.stdout:
        print(line.strip())

    return jsonify({'status': 'Training started, check the terminal for logs.'})

def generate_frames():
    """Capture video frames and save images if capturing is enabled."""
    global open_counter, closed_counter, capture, door_state

    while True:
        success, frame = camera.read()
        if not success:
            break

        if capture:
            if door_state == 'open':
                open_counter += 1
                cv2.imwrite(f"data/door_open/open_{open_counter}.jpg", frame)
            elif door_state == 'closed':
                closed_counter += 1
                cv2.imwrite(f"data/door_closed/closed_{closed_counter}.jpg", frame)
            print(f"Captured {door_state} door image")

        # Encode the frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Default video feed (door detection mode)."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Door detection mode video stream
@app.route('/door_detection_feed')
def door_detection_feed():
    """Door Detection Mode video feed."""
    return Response(generate_door_detection_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Regular YOLOv8 mode video stream
@app.route('/regular_yolo_feed')
def regular_yolo_feed():
    """Regular YOLOv8 Mode video feed."""
    return Response(generate_regular_yolo_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_door_detection_frames():
    """Generate frames using the trained door detection model."""
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run the door detection model
        results = door_model(frame)
        annotated_frame = results[0].plot()

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_regular_yolo_frames():
    """Generate frames using the general YOLOv8 model."""
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run the general YOLOv8 model
        results = yolo_model(frame)
        annotated_frame = results[0].plot()

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
