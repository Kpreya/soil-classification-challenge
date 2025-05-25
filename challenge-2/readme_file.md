# Soil Classification Challenge

This project implements a deep learning solution for soil image classification using PyTorch and computer vision techniques.

## Project Structure

```
challenge-1/
├── data/
│   └── download.sh          # Script to download dataset from Kaggle
├── docs/cards/
│   ├── architecture.png     # Model architecture diagram
│   └── ml-metrics.json     # Model performance metrics
├── notebooks/
│   ├── inference.ipynb     # Inference notebook
│   └── training.ipynb      # Training notebook
├── src/
│   ├── preprocessing.py    # Data preprocessing and augmentation
│   └── postprocessing.py   # Results analysis and metrics
├── requirements.txt        # Python dependencies
├── LICENSE                 # Project license
└── README.md              # This file
```

## Model Performance

Our soil classification model achieves excellent performance on the validation set:

- **Overall Accuracy**: 97.48%
- **F1-Score**: 98.36%
- **Precision**: 97.17%
- **Recall**: 99.59%

### Confusion Matrix
```
                Predicted
Actual          Non-Soil  Soil
Non-Soil           66      7
Soil                1    244
```

### Per-Class Performance
- **Non-Soil**: Precision: 98.51%, Recall: 90.41%, F1: 94.29%
- **Soil**: Precision: 97.21%, Recall: 99.59%, F1: 98.39%

## Key Features

### 1. Advanced Data Preprocessing
- **Image Quality Assessment**: Automatic evaluation of blur, brightness, and contrast
- **Image Enhancement**: Noise reduction and contrast improvement for low-quality images
- **Synthetic Negative Generation**: Creates non-soil images from soil images using various transformations

### 2. Robust Architecture
- **Transfer Learning**: ResNet50 backbone pre-trained on ImageNet
- **Enhanced Classifier Head**: Multi-layer classifier with dropout and batch normalization
- **Fallback Architecture**: Custom CNN if pre-trained model loading fails

### 3. Comprehensive Training Strategy
- **Data Augmentation**: Random crops, flips, rotations, color jittering
- **Synthetic Negatives**: Addresses class imbalance by generating artificial negative samples
- **Early Stopping**: Prevents overfitting using F1-score monitoring
- **Learning Rate Scheduling**: OneCycleLR for optimal convergence

### 4. Quality Enhancements
- **Image Processing**: Automatic enhancement of poor-quality images
- **Noise Reduction**: Advanced denoising techniques
- **Quality Scoring**: Automatic assessment and filtering of low-quality samples

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd challenge-1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
cd data
bash download.sh
```

## Usage

### Training and Inference
```bash
python main.py --mode both --data-dir /path/to/data --output-dir /path/to/output
```

### Training Only
```bash
python main.py --mode train --data-dir /path/to/data --output-dir /path/to/output
```

### Inference Only
```bash
python main.py --mode inference --data-dir /path/to/data --output-dir /path/to/output
```

## File Descriptions

### Core Components

- **`src/preprocessing.py`**: Contains data preprocessing classes and functions:
  - `ImageProcessor`: Quality assessment and enhancement
  - `NegativeGenerator`: Synthetic negative sample generation
  - `SoilDataset`: Custom PyTorch dataset with augmentation support

- **`src/postprocessing.py`**: Results analysis and metrics calculation:
  - Comprehensive metrics calculation
  - Performance analysis and interpretation
  - JSON export functionality

- **`main.py`**: Main training and inference script with command-line interface

### Notebooks

- **`notebooks/training.ipynb`**: Interactive training notebook with visualizations
- **`notebooks/inference.ipynb`**: Inference and results analysis notebook

### Documentation

- **`docs/cards/ml-metrics.json`**: Detailed model performance metrics
- **`docs/cards/architecture.png`**: Model architecture visualization

## Model Architecture

The model uses a ResNet50 backbone with an enhanced classifier head:

1. **Backbone**: ResNet50 pre-trained on ImageNet
2. **Feature Extraction**: Convolutional layers (partially frozen)
3. **Classifier Head**: 
   - Dropout (0.3) → Linear (2048→512) → BatchNorm → ReLU
   - Dropout (0.4) → Linear (512→256) → BatchNorm → ReLU  
   - Dropout (0.2) → Linear (256→2)

## Data Augmentation Strategy

### Training Augmentations
- Random resized crops (224×224)
- Random horizontal/vertical flips
- Random rotations (±20°)
- Color jittering (brightness, contrast, saturation, hue)
- Random grayscale conversion
- Random erasing

### Synthetic Negative Generation
- Abstract pattern creation
- Heavy blurring to destroy texture
- Noise pattern overlay
- Geometric shape insertion
- Extreme color distortion
- Texture removal and replacement

## Results Interpretation

The model demonstrates excellent performance with:

- **High Recall for Soil (99.59%)**: Very few soil images are missed
- **Good Precision for Both Classes**: Low false positive rates
- **Balanced Performance**: Works well for both soil and non-soil detection
- **Robust Generalization**: Strong performance on validation set

The slight bias toward classifying images as soil (7 false positives vs 1 false negative) is acceptable given the application context where missing soil samples might be more problematic than occasional false alarms.

## Future Improvements

1. **Ensemble Methods**: Combine multiple models for better robustness
2. **Advanced Augmentation**: Implement more sophisticated data augmentation techniques
3. **Active Learning**: Iteratively improve the model with human feedback
4. **Model Compression**: Optimize for deployment in resource-constrained environments
5. **Multi-Scale Analysis**: Process images at multiple resolutions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision for pre-trained models and transforms
- scikit-learn for evaluation metrics
- OpenCV for image processing capabilities