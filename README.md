# Face Mask Detection System

A real-time face mask detection system using deep learning and computer vision. This project uses transfer learning with ResNet50 to classify whether a person is wearing a mask or not.

## Overview

This project implements a complete pipeline for face mask detection:
1. Data Collection - Capture images from webcam or import from Kaggle
2. Data Preprocessing - Augmentation and normalization
3. Model Training - Fine-tune ResNet50 with transfer learning
4. Evaluation - Generate performance metrics
5. Real-time Detection - Live webcam inference

## Features

- Transfer learning with pretrained ResNet50
- Data augmentation for improved generalization
- Real-time face detection and classification
- Comprehensive evaluation metrics (confusion matrix, classification reports, training curves)
- Simple interface with visual feedback

## Requirements

### Hardware
- Webcam (for data collection and real-time detection)
- GPU recommended but not required
- At least 4GB RAM

### Software
- Python 3.7 or higher
- CUDA (optional, for GPU acceleration)

### Python Libraries
```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
pillow>=8.0.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
kaggle>=1.5.16
```

## Installation

### Step 1: Clone or Download the Repository
```bash
git clone https://github.com/yourusername/mask-detection.git
cd mask-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

## Dataset Preparation

### Option 1: Download from Kaggle (Recommended)

This project uses the Face Mask 12K Images Dataset from Kaggle.

**Setup Kaggle API:**
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section and click "Create New API Token"
3. Place the downloaded `kaggle.json` file in:
   - Windows: `C:\Users\<username>\.kaggle\`
   - Mac/Linux: `~/.kaggle/`
4. On Mac/Linux, set permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Download and organize dataset:**
```bash
python dataset_import.py
```

This will download approximately 12,000 images (500MB) and organize them into the required structure.

### Option 2: Collect Your Own Dataset

1. Prepare for collection with good lighting and a clear background
2. Run the data collection script:
   ```python
   from mask_detector import WebcamDataCollector
   
   collector = WebcamDataCollector()
   collector.collect_images(category='without_mask', num_images=100)
   collector.collect_images(category='with_mask', num_images=100)
   ```
3. Press SPACE to capture images, ESC to finish

### Dataset Structure

The organized dataset should have this structure:
```
mask_dataset/
├── with_mask/
│   ├── img_0.jpg
│   ├── img_1.jpg
│   └── ...
└── without_mask/
    ├── img_0.jpg
    ├── img_1.jpg
    └── ...
```

## Project Structure

```
mask-detection/
├── mask_detection.py          # Main training script
├── dataset_import.py           # Kaggle dataset downloader
├── run_webcam_detection.py     # Real-time detection script
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── mask_dataset/              # Dataset directory
│   ├── with_mask/
│   └── without_mask/
├── best_mask_detector.pth     # Saved model weights (after training)
├── confusion_matrix.png       # Evaluation results
└── training_history.png       # Training curves
```

## Usage

### Training the Model

Run the training script:
```bash
python mask_detection.py
```

The script will:
- Load the dataset from `mask_dataset/`
- Split into train/validation/test sets (60/20/20)
- Train ResNet50 for 10 epochs
- Save the best model as `best_mask_detector.pth`
- Generate evaluation metrics and visualizations

Training time varies based on hardware:
- CPU: 2-3 hours for full 12K dataset
- GPU: 15-30 minutes for full 12K dataset

### Real-time Detection

After training, run the webcam detection:
```bash
python run_webcam_detection.py
```

Controls:
- Press 'Q' to quit
- Press 'S' to save screenshot

The system will display:
- Green box around faces wearing masks
- Red box around faces without masks
- Confidence percentage for each detection
- FPS counter

## How It Works

### Data Preprocessing
- All images resized to 224x224 pixels
- Training augmentation: random horizontal flip, rotation, and color jitter
- Normalization using ImageNet mean and standard deviation

### Model Architecture

Base model: ResNet50 pretrained on ImageNet
```
Input (224x224x3)
    |
ResNet50 Backbone (early layers frozen)
    |
Global Average Pooling
    |
FC Layer (2048 -> 512) + ReLU + Dropout(0.5)
    |
FC Layer (512 -> 2)
    |
Softmax
    |
Output [P(No Mask), P(Mask)]
```

### Training Strategy
- Transfer learning approach: freeze early layers, fine-tune last 20 layers
- Optimizer: Adam with learning rate 0.0001
- Loss function: Cross-Entropy Loss
- Learning rate schedule: Step decay (gamma=0.1 every 5 epochs)

### Inference Pipeline
```
Webcam Frame -> Face Detection (Haar Cascade) -> Face Extraction -> 
Preprocessing -> Model Prediction -> Display Result
```

## Results

### Performance Metrics

With the 12K dataset, the model achieves:
- Training Accuracy: 95-99%
- Validation Accuracy: 93-97%
- Test Accuracy: 93-97%

Sample classification report:
```
                    precision   recall    f1-score   support
    Without Mask     0.999      0.999     0.999      2364
    With Mask        0.999      0.999     0.999      2353

        accuracy                          0.999      4717
    macro avg        0.999      0.999     0.999      4717
    weighted avg     0.999      0.999     0.999      4717
```

### Evaluation Outputs

The training script generates:
1. Classification report with precision, recall, and F1-score
2. Confusion matrix visualization (`confusion_matrix.png`)
3. Training history plots (`training_history.png`)

## Troubleshooting

### Webcam Not Detected
Try different camera indices in `run_webcam_detection.py`:
```python
CAMERA_INDEX = 1  # Change from 0 to 1, 2, etc.
```

### CUDA Out of Memory
Reduce batch size in `mask_detection.py`:
```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

### Face Detection Issues
- Ensure adequate lighting
- Face the camera directly
- Adjust detection parameters if needed:
```python
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
```

### Slow Training
- Use GPU if available (10-20x speedup)
- Reduce dataset size for testing:
  - Set `USE_SUBSET = True` in the training script
  - Adjust `SUBSET_PERCENTAGE` to use less data

## Configuration

Key hyperparameters that can be adjusted:

```python
# Training
num_epochs = 10          # Number of training epochs
batch_size = 16          # Batch size
learning_rate = 0.0001   # Initial learning rate

# Data Split
test_size = 0.2          # 20% for testing
val_size = 0.2           # 20% of training for validation

# Model
dropout = 0.5            # Dropout rate
num_classes = 2          # Binary classification
```

## Future Improvements

- Multi-class classification (proper mask wearing, improper mask wearing, no mask)
- Mobile deployment using TensorFlow Lite or ONNX
- Web interface using Flask or FastAPI
- Support for video file input
- Model compression and quantization
- Distance detection for social distancing monitoring

## References

- ResNet50 architecture: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- Transfer learning approach based on standard practices in computer vision
- OpenCV library for face detection and image processing
- PyTorch framework for deep learning implementation

## License

This project is licensed under the MIT License.

## Acknowledgments

- Kaggle for providing the Face Mask 12K Images Dataset
- PyTorch team for the deep learning framework
- OpenCV community for computer vision tools
- Original ResNet paper authors for the architecture