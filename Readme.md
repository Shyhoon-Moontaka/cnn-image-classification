# CNN Image Classification

A Convolutional Neural Network (CNN) implementation for geometric shapes classification using PyTorch.

## Project Overview

This project implements a deep learning solution for classifying geometric shapes (circles, squares, and triangles) using a Convolutional Neural Network. The model is trained on a custom dataset of shape images and achieves high accuracy in shape recognition tasks via PyTorch.

### Key Features

- **Shape Classification**: Automatically identifies circles, squares, and triangles
- **High Accuracy**: Optimized CNN architecture for reliable predictions
- **Custom Dataset**: Trained on carefully curated shape images
- **Model Persistence**: Pre-trained model ready for deployment

## Dataset

The dataset contains labeled images of three geometric shapes:

### Dataset Structure
```
dataset/
├── circle/          # Circle shape images
├── square/          # Square shape images
└── triangle/        # Triangle shape images
```

### Dataset Statistics

| Shape | Training Images | Format |
|-------|----------------|--------|
| Circle | 3 | PNG |
| Square | 4 | PNG |
| Triangle | 3 | PNG |
| **Total** | **10** | **PNG** |

## Model Architecture

The CNN architecture is designed for optimal shape classification performance:

- **Input Layer**: Accepts RGB images of variable sizes
- **Convolutional Layers**: Feature extraction using multiple filters
- **Pooling Layers**: Dimensionality reduction while preserving features
- **Fully Connected Layers**: Classification with softmax activation
- **Output**: 3 classes (Circle, Square, Triangle)

## Model Performance

### Training Results

<img width="981" height="451" alt="image" src="https://github.com/user-attachments/assets/f8b87b03-1294-4d15-ad98-37c9c6504b37" />

###Confusion Matrix

<img width="515" height="435" alt="image" src="https://github.com/user-attachments/assets/c9e2ae7b-f0f1-429a-9fdc-1b4790a82151" />

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cnn-shape-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

```python
# Run the training notebook
jupyter notebook 210105.ipynb
```

### Loading the Pre-trained Model

```python
import torch
import torch.nn as nn

# Load the trained model
model = torch.load('model/210105.pth')
model.eval()

# Make predictions
# Your prediction code here
```

### Making Predictions

```python
# Example prediction code
def predict_shape(image_path):
    # Load and preprocess image
    # Run through model
    # Return prediction
    pass
```

## File Structure

```
├── 210105.ipynb          # Main training notebook
├── Readme.md             # Project documentation
├── dataset/              # Training dataset
│   ├── circle/           # Circle images
│   ├── square/           # Square images
│   └── triangle/         # Triangle images
└── model/                # Trained models
    └── 210105.pth        # PyTorch model file
```

## Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
jupyter>=1.0.0
pillow>=8.3.0
```

## Results

The model demonstrates excellent performance on the shape classification task:

- **Accuracy**: 87.5%+ on test set
- **Training Time**: Optimized for fast convergence
- **Model Size**: Lightweight for deployment

### Sample Predictions

<img width="1189" height="1140" alt="image" src="https://github.com/user-attachments/assets/11b68fc5-3b76-4986-96f6-a049fbebd13b" />

## Visual Error Analysis

<img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/766499c4-00eb-4efe-b601-440b47e888d8" />

## Author

- **Shyhoon Moontaka** - *Initial work* - https://github.com/Shyhoon-Moontaka


**Note**: This project uses placeholder images for demonstration. Replace with actual images for production use.
