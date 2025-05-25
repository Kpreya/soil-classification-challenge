# Soil Classification with Deep Learning

A deep learning solution for classifying soil types from images using PyTorch. This project implements a CNN-based classifier that can identify four different soil types: Alluvial soil, Black Soil, Clay soil, and Red soil.

## Project Overview

This project provides an end-to-end machine learning pipeline for soil classification, achieving **92% accuracy** on the validation set. The solution uses transfer learning with ResNet50 as the backbone architecture, with custom preprocessing and data augmentation techniques.

##  Model Performance

| Soil Type     | Precision | Recall | F1-Score | Support |
|---------------|-----------|---------|----------|---------|
| Alluvial soil | 0.99      | 0.88    | 0.93     | 106     |
| Black Soil    | 0.88      | 0.93    | 0.91     | 46      |
| Clay soil     | 0.78      | 0.97    | 0.87     | 40      |
| Red soil      | 0.98      | 0.96    | 0.97     | 53      |

**Overall Accuracy**: 92%  
**Macro Average F1-Score**: 0.92  
**Weighted Average F1-Score**: 0.92

## Project Structure

```
challenge-1/
├── data/                          # Data directory (downloaded from Kaggle)
│   └── download.sh               # Script to download data
├── docs/cards/                   # Documentation and model cards
│   ├── architecture.png          # Model architecture diagram
│   └── ml-metrics.json          # Detailed performance metrics
├── notebooks/                    # Jupyter notebooks
│   ├── inference.ipynb          # Inference and prediction examples
│   └── training.ipynb           # Complete training pipeline
├── src/                         # Source code
│   ├── postprocessing.py       # Results processing and visualization
│   └── preprocessing.py        # Data loading and preprocessing
├── LICENSE                      # License file
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd challenge-1

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download data from Kaggle (requires Kaggle API setup)
cd data
bash download.sh
```

### 3. Training

```python
# Run the complete training pipeline
python notebooks/training.py
```

### 4. Inference

```python
# Use the trained model for predictions
from src.inference_utils import SoilClassificationInference

classifier = SoilClassificationInference('models/best_model.pth')
result = classifier.predict('path/to/soil_image.jpg')
print(result)
```

##  Model Architecture

The model uses a **ResNet50** backbone with transfer learning:

- **Input**: RGB images (224×224×3)
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Feature Extraction**: Convolutional layers (frozen early layers)
- **Classifier Head**: 
  - Linear(2048 → 512)
  - ReLU + Dropout(0.3)
  - Linear(512 → 4)
- **Output**: 4 soil type classes

### Data Augmentation
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness/contrast ±0.2)
- Standard ImageNet normalization

##  Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Early Stopping**: Patience of 5 epochs
- **LR Scheduler**: ReduceLROnPlateau
- **Training/Validation Split**: 80/20

##  Usage Examples

### Single Image Prediction

```python
from src.preprocessing import SoilDataset
from src.training import SoilClassifier
import torch
from PIL import Image
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoilClassifier()
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Make prediction
image = Image.open('soil_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, 1).item()

class_names = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]
print(f"Predicted soil type: {class_names[predicted_class]}")
```

### Batch Processing

```python
from torch.utils.data import DataLoader
from src.preprocessing import SoilDataset

# Create dataset
test_dataset = SoilDataset(test_df, test_dir, transform=transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Generate predictions
predictions = []
model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        predictions.extend(preds.cpu().numpy())
```

##  Results and Visualizations

The training pipeline generates several outputs:

- **Confusion Matrix**: Visual representation of classification performance
- **Training History**: Loss and accuracy curves over epochs
- **Classification Report**: Detailed per-class metrics
- **Model Architecture**: Detailed model structure information

## Development



## 🔍Model Insights

### Strong Performance
- **Alluvial soil**: Highest precision (0.99) - very few false positives
- **Red soil**: Excellent overall performance (F1: 0.97)
- **Black Soil**: Good balance of precision and recall

### Areas for Improvement  
- **Clay soil**: Lower precision (0.78) suggests some confusion with other types
- Consider collecting more Clay soil samples to improve balance




## Acknowledgments

- Dataset provided by Kaggle Soil Classification Challenge
- PyTorch team for the excellent deep learning framework
- ResNet50 architecture from "Deep Residual Learning for Image Recognition"


