"""
API Client Example for Parkinson's Disease Detection
=====================================================

HOW TO USE THIS FILE:
--------------------
1. Start the API server first:
   cd deployment
   python app.py

2. Then run this client:
   python api_client.py

3. Or use the web interface:
   Open http://localhost:5000 in your browser

4. Or import in your own Python code:
   from api_client import predict_single, generate_sample_data
"""

import requests
import json
import base64
import numpy as np
import os


API_BASE_URL = "http://localhost:5000"


def check_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "error": "Could not connect to API"}


def get_model_info():
    """Get information about the loaded model."""
    response = requests.get(f"{API_BASE_URL}/model-info")
    return response.json()


def predict_single(voice_features, gait_signal, image_path_or_base64):
    """
    Make a prediction for a single patient.
    
    Parameters:
    -----------
    voice_features: list
        List of 22 voice feature values
    gait_signal: list
        List of gait signal values (will be resampled to 512)
    image_path_or_base64: str
        Either a file path to the handwriting image or base64 encoded string
    
    Returns:
    --------
    dict: Prediction result
    """
    # Prepare image data
    if image_path_or_base64.endswith(('.png', '.jpg', '.jpeg')):
        # It's a file path, read and encode as base64
        with open(image_path_or_base64, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
    else:
        image_data = image_path_or_base64
    
    payload = {
        "voice": voice_features,
        "gait": gait_signal,
        "image": image_data
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()


def predict_batch(patients):
    """
    Make predictions for multiple patients.
    
    Parameters:
    -----------
    patients: list of dict
        Each dict should have: id, voice, gait, image
    
    Returns:
    --------
    dict: Batch prediction results
    """
    response = requests.post(
        f"{API_BASE_URL}/batch-predict",
        json={"patients": patients},
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()


def generate_sample_data():
    """Generate sample data for testing."""
    # Sample voice features (22 features from UCI Parkinson's dataset)
    voice_features = [
        119.992, 157.302, 74.997, 0.00784, 0.00007,
        0.00370, 0.00554, 0.01109, 0.04374, 0.426,
        0.02182, 0.02254, 0.03052, 0.06545, 0.02211,
        21.033, 0.414783, 0.815285, -4.813031, 0.266482,
        2.301442, 0.284654
    ]
    
    # Sample gait signal (random for demo)
    gait_signal = np.random.randn(1000).tolist()
    
    return voice_features, gait_signal


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Parkinson's Detection API Client")
    print("=" * 60)
    
    # 1. Check API health
    print("\n1. Checking API health...")
    health = check_health()
    print(f"   Status: {health}")
    
    if health.get('status') != 'healthy':
        print("\n" + "=" * 60)
        print("⚠️  API IS NOT RUNNING!")
        print("=" * 60)
        print("""
To start the API server:

    Step 1: Open a terminal/command prompt
    Step 2: Navigate to the deployment folder:
            cd deployment
    Step 3: Run the API server:
            python app.py
    Step 4: Keep the server running and try this script again

Alternatively, use the web interface:
    - Start the server (python app.py)
    - Open http://localhost:5000 in your browser
        """)
        exit(1)
    
    # 2. Get model info
    print("\n2. Getting model information...")
    model_info = get_model_info()
    print(f"   Model: {model_info.get('name')}")
    print(f"   Modalities: {model_info.get('modalities')}")
    print(f"   Threshold: {model_info.get('optimal_threshold')}")
    
    # 3. Prepare sample data
    print("\n3. Preparing sample data...")
    voice, gait = generate_sample_data()
    print(f"   Voice features: {len(voice)} values")
    print(f"   Gait signal: {len(gait)} values")
    
    # 4. Try to make a prediction with a real image
    print("\n4. Looking for a sample image...")
    
    # Try to find a sample image from the drawing dataset
    sample_image_paths = [
        "../drawingdataset/spiral/training/healthy/V01HE01.png",
        "../drawingdataset/wave/training/healthy/V01HO01.png",
        "../drawingdataset/drawings/spiral/training/healthy/V01HE01.png",
    ]
    
    image_path = None
    for path in sample_image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path:
        print(f"   Found sample image: {image_path}")
        print("\n5. Making prediction...")
        
        result = predict_single(voice, gait, image_path)
        
        print("\n" + "=" * 60)
        print("🎯 PREDICTION RESULT")
        print("=" * 60)
        print(f"   Prediction:  {result.get('prediction')}")
        print(f"   Probability: {result.get('probability', 0):.2%}")
        print(f"   Confidence:  {result.get('confidence')}")
        print(f"   Threshold:   {result.get('threshold_used')}")
    else:
        print("   No sample image found in the dataset folder")
        print("   To make a real prediction, provide an image path:")
        print("   result = predict_single(voice, gait, 'path/to/image.png')")
    
    print("\n" + "=" * 60)
    print("📖 HOW TO USE IN YOUR OWN CODE")
    print("=" * 60)
    print("""
    # Import the functions
    from api_client import predict_single, generate_sample_data
    
    # Get sample data
    voice, gait = generate_sample_data()
    
    # Make a prediction (provide your own image)
    result = predict_single(
        voice_features=voice,
        gait_signal=gait,
        image_path_or_base64="path/to/spiral_drawing.png"
    )
    
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.2%}")
    """)
    
    print("\n" + "=" * 60)
    print("🌐 WEB INTERFACE")
    print("=" * 60)
    print("""
    For an easier experience, use the web interface:
    
    1. Make sure the API is running (python app.py)
    2. Open your browser to: http://localhost:5000
    3. Upload your data through the web form
    4. Get instant predictions!
    """)
