import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import cv2
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable CORS for web applications

# Configuration
MODEL_PATH = '../ImprovedModels/improved_fusion_model.h5'
VOICE_SCALER_PATH = '../ImprovedModels/voice_scaler.pkl'
GAIT_SCALER_PATH = '../ImprovedModels/gait_scaler.pkl'
OPTIMAL_THRESHOLD_PATH = '../ImprovedModels/optimal_threshold.txt'

# Global variables for model and scalers
model = None
voice_scaler = None
gait_scaler = None
optimal_threshold = 0.5


def load_resources():
    """Load model and preprocessing resources."""
    global model, voice_scaler, gait_scaler, optimal_threshold
    
    print("Loading Parkinson's Detection Model...")
    
    # Load the trained model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Load voice scaler
    if os.path.exists(VOICE_SCALER_PATH):
        with open(VOICE_SCALER_PATH, 'rb') as f:
            voice_scaler = pickle.load(f)
        print(f"✓ Voice scaler loaded")
    else:
        print("⚠ Voice scaler not found, using per-sample normalization")
    
    # Load gait scaler
    if os.path.exists(GAIT_SCALER_PATH):
        with open(GAIT_SCALER_PATH, 'rb') as f:
            gait_scaler = pickle.load(f)
        print(f"✓ Gait scaler loaded")
    else:
        print("⚠ Gait scaler not found, using per-sample normalization")
    
    # Load optimal threshold
    if os.path.exists(OPTIMAL_THRESHOLD_PATH):
        with open(OPTIMAL_THRESHOLD_PATH, 'r') as f:
            for line in f:
                if 'Optimal Threshold' in line:
                    optimal_threshold = float(line.split(':')[1].strip())
        print(f"✓ Optimal threshold: {optimal_threshold}")
    
    print("✓ All resources loaded successfully!")


def preprocess_voice(voice_features):
    """Preprocess voice features for model input."""
    global voice_scaler
    voice_array = np.array(voice_features, dtype=np.float32).reshape(1, -1)
    
    # Use trained scaler if available, otherwise per-sample normalization
    if voice_scaler is not None:
        voice_array = voice_scaler.transform(voice_array)
    else:
        voice_array = (voice_array - np.mean(voice_array)) / (np.std(voice_array) + 1e-8)
    
    # Reshape for 1D-CNN: (1, features, 1)
    voice_array = voice_array.reshape(1, -1, 1)
    
    return voice_array


def preprocess_gait(gait_signal, target_length=512):
    """Preprocess gait signal for model input."""
    gait_array = np.array(gait_signal, dtype=np.float32)
    
    # Resample to target length
    if len(gait_array) > target_length:
        indices = np.linspace(0, len(gait_array)-1, target_length).astype(int)
        gait_array = gait_array[indices]
    else:
        padded = np.zeros(target_length)
        padded[:len(gait_array)] = gait_array
        gait_array = padded
    
    # Use trained scaler if available, otherwise per-sample normalization
    global gait_scaler
    if gait_scaler is not None:
        gait_array = gait_scaler.transform(gait_array.reshape(-1, 1)).flatten()
    else:
        gait_array = (gait_array - np.mean(gait_array)) / (np.std(gait_array) + 1e-8)
    
    # Reshape for 1D-CNN: (1, timesteps, 1)
    gait_array = gait_array.reshape(1, target_length, 1)
    
    return gait_array


def preprocess_image(image_data, img_size=128):
    """
    Preprocess handwriting image for model input.
    
    Accepts either:
    - Base64 encoded image string
    - Path to image file
    - Raw image array
    """
    if isinstance(image_data, str):
        if os.path.exists(image_data):
            # It's a file path
            img = cv2.imread(image_data, cv2.IMREAD_GRAYSCALE)
        else:
            # Assume base64 encoded
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    else:
        img = np.array(image_data, dtype=np.uint8)
    
    if img is None:
        raise ValueError("Could not load image")
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Reshape for 2D-CNN: (1, height, width, 1)
    img = img.reshape(1, img_size, img_size, 1)
    
    return img


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'name': 'Multimodal Parkinson Detection CNN',
        'modalities': ['voice', 'gait', 'handwriting'],
        'input_shapes': {
            'voice': '(22, 1)',
            'gait': '(512, 1)',
            'handwriting': '(128, 128, 1)'
        },
        'optimal_threshold': optimal_threshold,
        'output': 'Probability of Parkinson\'s Disease (0-1)'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction using multimodal data.
    
    Request JSON format:
    {
        "voice": [list of 22 voice features],
        "gait": [list of gait signal values],
        "image": "base64_encoded_image_string" OR file path
    }
    
    Response:
    {
        "prediction": "Parkinson's" or "Healthy",
        "probability": 0.85,
        "confidence": "High/Medium/Low",
        "threshold_used": 0.5
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['voice', 'gait', 'image']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'required': required_fields
            }), 400
        
        # Preprocess inputs
        voice_input = preprocess_voice(data['voice'])
        gait_input = preprocess_gait(data['gait'])
        image_input = preprocess_image(data['image'])
        
        # Make prediction
        probability = model.predict([voice_input, gait_input, image_input], verbose=0)[0][0]
        probability = float(probability)
        
        # Apply optimal threshold
        prediction = "Parkinson's" if probability >= optimal_threshold else "Healthy"
        
        # Determine confidence level
        distance_from_threshold = abs(probability - optimal_threshold)
        if distance_from_threshold > 0.3:
            confidence = "High"
        elif distance_from_threshold > 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 4),
            'confidence': confidence,
            'threshold_used': optimal_threshold,
            'raw_score': round(probability, 6)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/predict/voice-only', methods=['POST'])
def predict_voice_only():
    """Predict using only voice features (single modality)."""
    return jsonify({'error': 'Single-modality prediction not yet implemented'}), 501


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple patients at once.
    
    Request JSON format:
    {
        "patients": [
            {"id": "patient_1", "voice": [...], "gait": [...], "image": "..."},
            {"id": "patient_2", "voice": [...], "gait": [...], "image": "..."}
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'patients' not in data:
            return jsonify({'error': 'Missing "patients" field'}), 400
        
        results = []
        for patient in data['patients']:
            try:
                voice_input = preprocess_voice(patient['voice'])
                gait_input = preprocess_gait(patient['gait'])
                image_input = preprocess_image(patient['image'])
                
                probability = model.predict([voice_input, gait_input, image_input], verbose=0)[0][0]
                probability = float(probability)
                
                results.append({
                    'patient_id': patient.get('id', 'unknown'),
                    'prediction': "Parkinson's" if probability >= optimal_threshold else "Healthy",
                    'probability': round(probability, 4),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'patient_id': patient.get('id', 'unknown'),
                    'error': str(e),
                    'status': 'failed'
                })
        
        return jsonify({
            'results': results,
            'total': len(results),
            'successful': sum(1 for r in results if r['status'] == 'success')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model and resources
    load_resources()
    
    # Run the Flask app
    print("\n" + "="*50)
    print("Parkinson's Detection API Running!")
    print("="*50)
    print("Endpoints:")
    print("  - GET  /health       - Health check")
    print("  - GET  /model-info   - Model information")
    print("  - POST /predict      - Single prediction")
    print("  - POST /batch-predict - Batch predictions")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)