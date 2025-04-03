from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2  # For real-time video capture
from io import BytesIO
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model globally
model_path = 'C:/Users/angelikadennise/Downloads/Bangus_/cnn_scoop.h5'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# List of class names (must match the model's output)
class_names = ["Aeromonas Hydrophila", "Normal Milkfish", "Vibrio Harveyi"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Milkfish disease detection API is running!"})

# Function to process image frames and make predictions
def process_frame(frame):
    img = Image.fromarray(frame)
    img = img.resize((300, 300))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    predicted_prob = predictions[0][predicted_class_index]
    return predicted_class, predicted_prob

# Video streaming for real-time detection (using OpenCV)
@app.route('/video_feed')
def video_feed():
    def generate():
        # Open webcam (or video file) for capturing frames
        video_capture = cv2.VideoCapture(0)  # Use `0` for the default webcam
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Process the captured frame and get prediction
            predicted_class, predicted_prob = process_frame(frame)

            # Annotate the frame with the prediction
            cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Probability: {predicted_prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Encode frame as JPEG and send to the client
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
        video_capture.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=["POST"])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the image from the request
        image = request.files['image']
        img = Image.open(image.stream).resize((300, 300))  # Resize to match model input
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        predicted_prob = predictions[0][predicted_class_index]

        # Return the prediction result
        return jsonify({
            'prediction': predicted_class,
            'probability': float(predicted_prob)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=["POST"])
def upload_image():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Read and preprocess the image
        image = request.files['image']
        img = Image.open(image.stream).resize((300, 300))  # Resize to match model input
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        predicted_prob = predictions[0][predicted_class_index]

        # Return the prediction result
        return jsonify({
            'prediction': predicted_class,
            'probability': float(predicted_prob)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

