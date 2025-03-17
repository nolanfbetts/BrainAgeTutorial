# BrainAgeTutorial
Project for AI in Health care course as part of a Master's Program at UT Austin. This project leverages the OASIS dataset to try and predict brain age of patients using a CNN on brain scan MRIs 

This project implements a deep learning model to predict brain age from structural MRI scans. The model uses a 3D Convolutional Neural Network (CNN) with residual blocks to analyze brain MRI data and estimate the chronological age of subjects.

## Overview

The model processes structural MRI scans from the OASIS longitudinal dataset and predicts the age of subjects. It includes data preprocessing, augmentation, and a sophisticated training pipeline with learning rate scheduling and early stopping.

### Key Features

- 3D CNN architecture with residual blocks
- Data augmentation for improved generalization
- Learning rate scheduling with warmup
- Early stopping to prevent overfitting
- Comprehensive training metrics visualization
- Support for multiple hardware devices (CPU, CUDA, Apple Silicon)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

The project expects the following data structure:
```
data/
├── OAS2_RAW_PART1/
│   └── [MRI ID]/
│       └── RAW/
│           └── mpr-*.nifti.img
└── OAS2_RAW_PART2/
    └── [MRI ID]/
        └── RAW/
            └── mpr-*.nifti.img
```

## Usage

1. Place your data in the appropriate directory structure
2. Run the training script:
```bash
python self_learning_tutorial.py
```

The script will:
- Load and preprocess the MRI scans
- Split the data into training and test sets
- Train the model with the specified configuration
- Generate training metrics plots
- Save the best model checkpoint

## Model Architecture

The model uses a 3D CNN with:
- Initial convolutional layer with batch normalization and dropout
- Three residual blocks with increasing channels (16→32→64→128)
- Global average pooling
- Fully connected layers with dropout

## Training Configuration

- Batch size: 8
- Initial learning rate: 0.001
- Learning rate warmup: 3 epochs
- Early stopping patience: 10 epochs
- Learning rate scheduler patience: 5 epochs
- Maximum epochs: 50

## Output

The training process generates:
- `best_brain_age_model.pth`: The best model checkpoint
- `training_metrics.png`: Visualization of training metrics including:
  - Loss curves
  - MAE and RMSE
  - Learning rate changes
  - Gradient norms

## Performance Metrics

The model is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Validation loss

## Hardware Support

The model automatically detects and uses the best available hardware:
- CUDA-capable NVIDIA GPUs
- Apple Silicon (MPS)
- CPU (fallback)
