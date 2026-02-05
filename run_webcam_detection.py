"""
Real-time Face Mask Detection using Webcam
Run this script after training to test your model in real-time
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

# ============================================================================
# MODEL DEFINITION (Same as training)
# ============================================================================

class MaskDetector(nn.Module):
    """ResNet50-based mask detector"""
    
    def __init__(self, num_classes=2):
        super(MaskDetector, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


# ============================================================================
# REAL-TIME DETECTION CLASS
# ============================================================================

class RealTimeMaskDetector:
    """Real-time mask detection using webcam"""
    
    def __init__(self, model_path='best_mask_detector.pth', device='cuda'):
        print("=" * 70)
        print("INITIALIZING MASK DETECTOR")
        print("=" * 70)
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        # Check if model file exists
        import os
        if not os.path.exists(model_path):
            print(f"\n❌ Error: Model file '{model_path}' not found!")
            print("Please train the model first by running: python mask_detection.py")
            sys.exit(1)
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = MaskDetector(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully!")
        
        # Face detector
        print("Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ Face detector loaded!")
        
        # Image transform (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Class names and colors
        self.class_names = ['No Mask', 'Mask']
        self.colors = [(0, 0, 255), (0, 255, 0)]  # Red for no mask, Green for mask
        
        print("\n" + "=" * 70)
        print("READY FOR DETECTION!")
        print("=" * 70)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("=" * 70)
    
    def detect(self, camera_index=0, show_fps=True):
        """
        Run real-time detection
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            show_fps: Whether to display FPS on screen
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("\n❌ Error: Could not open webcam!")
            print("Troubleshooting:")
            print("  - Make sure your webcam is connected")
            print("  - Try different camera index: detector.detect(camera_index=1)")
            print("  - Check if another application is using the webcam")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎥 Webcam started! Show your face to the camera...")
        
        # FPS calculation
        import time
        prev_time = time.time()
        fps = 0
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n❌ Error: Failed to grab frame")
                break
            
            # Calculate FPS
            if show_fps:
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract and preprocess face
                face = frame[y:y+h, x:x+w]
                
                try:
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = self.model(face_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        pred_class = predicted.item()
                        conf = confidence.item()
                    
                    # Draw results
                    label = f"{self.class_names[pred_class]}: {conf*100:.1f}%"
                    color = self.colors[pred_class]
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(
                        frame, 
                        (x, y - label_size[1] - 10), 
                        (x + label_size[0], y), 
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame, 
                        label, 
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (255, 255, 255), 
                        2
                    )
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Display FPS
            if show_fps:
                cv2.putText(
                    frame, 
                    f"FPS: {fps:.1f}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
            
            # Display face count
            face_count_text = f"Faces: {len(faces)}"
            cv2.putText(
                frame, 
                face_count_text, 
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Display instructions
            cv2.putText(
                frame,
                "Press 'q' to quit | 's' to screenshot",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Show frame
            cv2.imshow('Face Mask Detection - Press Q to Quit', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q') or key == 27:  # q or ESC
                print("\n👋 Exiting...")
                break
            
            # Save screenshot
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Detection stopped.")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the detector"""
    
    print("\n" + "=" * 70)
    print("FACE MASK DETECTION - REAL-TIME WEBCAM")
    print("=" * 70)
    
    # Configuration
    MODEL_PATH = 'best_mask_detector.pth'  # Path to your trained model
    DEVICE = 'cuda'  # Use 'cuda' for GPU, 'cpu' for CPU
    CAMERA_INDEX = 0  # 0 for default webcam, try 1, 2 if it doesn't work
    SHOW_FPS = True  # Show FPS counter
    
    # Create detector
    try:
        detector = RealTimeMaskDetector(model_path=MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"\n❌ Error initializing detector: {e}")
        return
    
    # Run detection
    try:
        detector.detect(camera_index=CAMERA_INDEX, show_fps=SHOW_FPS)
    except KeyboardInterrupt:
        print("\n\n⚠ Detection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()