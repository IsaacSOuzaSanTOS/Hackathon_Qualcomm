#!/usr/bin/env python3
"""
Test script for running inference with the ONNX yawn classifier model.
"""

import onnxruntime as ort
import numpy as np
import cv2
import mediapipe as mp
import os

def centralizar_normalizar(pontos):
    """Normalize mouth landmarks using the same method as training."""
    pontos = np.array(pontos)
    if pontos.ndim == 1:
        pontos = pontos.reshape(-1, 2)
    
    centroide = pontos.mean(axis=0)
    pontos_centralizados = pontos - centroide
    
    norma = np.linalg.norm(pontos_centralizados, axis=1).max()
    if norma == 0:
        norma = 1
    pontos_normalizados = pontos_centralizados / norma
    
    return np.array(pontos_normalizados)

def extract_mouth_landmarks(image):
    """Extract mouth landmarks from an image using MediaPipe."""
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    # Mouth landmarks indices (same as training)
    mouth_landmarks = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
        14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
        312, 13, 82, 81, 42, 183, 78, 50, 280, 152, 377, 400, 164, 393
    ]
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    coords = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = image.shape
            for idx in mouth_landmarks:
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * width)
                y = int(lm.y * height)
                coords += [x, y]
    
    face_mesh.close()
    return coords

def load_and_prepare_image(image_path):
    """Load an image and extract normalized mouth landmarks."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Extract mouth landmarks
    coords = extract_mouth_landmarks(image)
    if not coords:
        raise ValueError(f"No face detected in image: {image_path}")
    
    # Normalize landmarks
    normalized_coords = centralizar_normalizar(coords)
    
    # Flatten and convert to float32
    return normalized_coords.flatten().astype(np.float32)

def run_inference(session, input_data):
    """Run inference with the ONNX model."""
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    results = session.run(None, {input_name: input_data})
    
    # Get logits
    logits = results[0][0]
    
    # Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Get predicted class
    predicted_class = np.argmax(probabilities)
    
    return predicted_class, probabilities

def main():
    # Load ONNX model
    model_path = "yawn_classifier.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"📦 Loading ONNX model from: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("✅ Model loaded successfully")
    
    # Test with sample images
    test_folders = {
        "test/no_yawn": "No Yawn",
        "test/yawn": "Yawn"
    }
    
    print("\n🔍 Running inference on test images...")
    print("=" * 60)
    
    for folder, label in test_folders.items():
        if not os.path.exists(folder):
            print(f"⚠️ Test folder not found: {folder}")
            continue
        
        print(f"\n📁 Testing images from: {folder}")
        files = os.listdir(folder)[:5]  # Test first 5 images from each folder
        
        for filename in files:
            image_path = os.path.join(folder, filename)
            try:
                # Process image
                input_data = load_and_prepare_image(image_path)
                
                # Run inference
                predicted_class, probabilities = run_inference(session, input_data)
                
                # Convert prediction to label
                predicted_label = "No Yawn" if predicted_class == 0 else "Yawn"  # Fix label mapping
                confidence = probabilities[predicted_class]
                
                # Print results
                print(f"\n🖼️  Image: {filename}")
                print(f"   Expected: {label}")
                print(f"   Predicted: {predicted_label} (confidence: {confidence:.2%})")
                print(f"   {'✅' if label == predicted_label else '❌'} " + 
                      f"{'Correct!' if label == predicted_label else 'Incorrect'}")
                
            except Exception as e:
                print(f"\n⚠️  Error processing {filename}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✨ Inference test complete!")

if __name__ == "__main__":
    main()
