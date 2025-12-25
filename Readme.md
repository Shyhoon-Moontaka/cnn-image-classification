# CNN Shape Classification Project

A Convolutional Neural Network (CNN) project for classifying geometric shapes including circles, squares, and triangles.

## Project Overview

This project implements a deep learning model using CNN architecture to automatically classify images of geometric shapes. The model is trained to distinguish between three different shape categories:
- Circle
- Square 
- Triangle

## Dataset

The dataset consists of PNG images organized in the following structure:

```
dataset/
├── circle/
│   ├── Circle_.png
│   ├── Circle_(1).png
│   └── Circle_(2).png
├── square/
│   ├── Square_.png
│   ├── Square_(1).png
│   ├── Square_(2).png
│   └── Square_(3).png
└── triangle/
    ├── Triangle_.png
    ├── Triangle_(1).png
    └── Triangle_(2).png
```

## Model

The trained CNN model is saved as `model/210105.pth`. The model was trained on the shape dataset and can be used for inference on new shape images.

## Files

- `210105.ipynb` - Jupyter notebook containing the complete implementation and training pipeline
- `model/210105.pth` - Trained CNN model weights
- `dataset/` - Directory containing all training and test images
- `Readme.md` - This documentation file

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- PIL (Pillow)
- Jupyter Notebook

## Usage

1. Open `210105.ipynb` in Jupyter Notebook
2. Run the cells to load the trained model or train a new one
3. Use the model to classify new shape images

## Model Architecture

The CNN architecture includes:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Softmax output layer for multi-class classification

## Training Details

- Model trained on: 2021-01-05 (based on filename)
- Classes: 3 (Circle, Square, Triangle)
- Input: RGB images
- Output: Shape classification probabilities

## Results

The model achieves classification accuracy on the test dataset. Specific performance metrics can be found in the Jupyter notebook.

## Future Improvements

- Increase dataset size with more diverse shape images
- Data augmentation techniques
- Experiment with different CNN architectures
- Add more shape categories
- Implement real-time shape detection

## Author

Created as part of a machine learning project for shape classification using convolutional neural networks.