"""
Face Mask Detection System using ResNet50
This system detects whether a person is wearing a mask using transfer learning
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ============================================================================
# PART 1: DATA COLLECTION
# ============================================================================

class WebcamDataCollector:
    """Collect images from webcam for dataset creation"""
    
    def __init__(self, save_dir='mask_dataset'):
        self.save_dir = save_dir
        os.makedirs(f"{save_dir}/with_mask", exist_ok=True)
        os.makedirs(f"{save_dir}/without_mask", exist_ok=True)
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def collect_images(self, category='with_mask', num_images=100):
        """
        Collect images from webcam
        
        Args:
            category: 'with_mask' or 'without_mask'
            num_images: number of images to collect
        """
        cap = cv2.VideoCapture(0)
        count = 0
        
        print(f"Collecting {num_images} images for '{category}'")
        print("Press SPACE to capture, ESC to exit")
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw rectangle around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display count
            cv2.putText(frame, f"Collected: {count}/{num_images}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Press SPACE to capture
            if key == ord(' ') and len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    filename = f"{self.save_dir}/{category}/img_{count}.jpg"
                    cv2.imwrite(filename, face)
                    count += 1
                    print(f"Captured image {count}")
                    break
            
            # Press ESC to exit
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collection complete! Saved {count} images.")


# ============================================================================
# PART 2: DATASET AND DATA LOADING
# ============================================================================

class MaskDataset(Dataset):
    """Custom dataset for mask detection"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def load_dataset(data_dir='mask_dataset'):
    """Load images and labels from directory"""
    image_paths = []
    labels = []
    
    # Load 'with_mask' images (label = 1)
    with_mask_dir = os.path.join(data_dir, 'with_mask')
    if os.path.exists(with_mask_dir):
        for img_name in os.listdir(with_mask_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(with_mask_dir, img_name))
                labels.append(1)
    
    # Load 'without_mask' images (label = 0)
    without_mask_dir = os.path.join(data_dir, 'without_mask')
    if os.path.exists(without_mask_dir):
        for img_name in os.listdir(without_mask_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(without_mask_dir, img_name))
                labels.append(0)
    
    return image_paths, labels


# ============================================================================
# PART 3: MODEL DEFINITION
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
# PART 4: TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """Train the mask detection model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_mask_detector.pth')
            print(f"Saved best model with accuracy: {best_acc:.4f}")
        
        scheduler.step()
    
    return model, history


# ============================================================================
# PART 5: EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and generate metrics"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Without Mask', 'With Mask']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Without Mask', 'With Mask'],
                yticklabels=['Without Mask', 'With Mask'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return all_preds, all_labels


# ============================================================================
# PART 6: REAL-TIME DETECTION
# ============================================================================

class RealTimeMaskDetector:
    """Real-time mask detection using webcam"""
    
    def __init__(self, model_path='best_mask_detector.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = MaskDetector(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['No Mask', 'Mask']
        self.colors = [(0, 0, 255), (0, 255, 0)]  # Red for no mask, Green for mask
    
    def detect(self):
        """Run real-time detection"""
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Extract and preprocess face
                face = frame[y:y+h, x:x+w]
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
                label = f"{self.class_names[pred_class]}: {conf:.2f}"
                color = self.colors[pred_class]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Mask Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# PART 7: MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Data Collection (Uncomment to collect data)
    # collector = WebcamDataCollector()
    # collector.collect_images(category='without_mask', num_images=50)
    # collector.collect_images(category='with_mask', num_images=50)
    
    # Step 2: Load dataset
    print("\nLoading dataset...")
    image_paths, labels = load_dataset('mask_dataset')
    print(f"Total images: {len(image_paths)}")
    print(f"With mask: {sum(labels)}, Without mask: {len(labels) - sum(labels)}")
    
    # Step 3: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Step 4: Create data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Step 5: Create datasets and dataloaders
    train_dataset = MaskDataset(X_train, y_train, transform=train_transform)
    val_dataset = MaskDataset(X_val, y_val, transform=test_transform)
    test_dataset = MaskDataset(X_test, y_test, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Step 6: Create and train model
    print("\nCreating model...")
    model = MaskDetector(num_classes=2).to(device)
    
    print("\nTraining model...")
    model, history = train_model(model, train_loader, val_loader, 
                                 num_epochs=10, device=device)
    
    # Step 7: Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_loader, device=device)
    
    # Step 8: Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("\nTraining complete! Model saved as 'best_mask_detector.pth'")
    
    # Step 9: Real-time detection (Uncomment to run)
    # print("\nStarting real-time detection...")
    # detector = RealTimeMaskDetector(model_path='best_mask_detector.pth', device=device)
    # detector.detect()


if __name__ == "__main__":
    main()