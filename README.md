# Lithology Classification with CNN

## Project Description

This project classifies lithology from drill core images into four categories (sandstone, limestone, shale, and garbage) using a Convolutional Neural Network (ResNeXt-50 architecture).

## Folder Structure
- `dataset/`: Contains the training, validation, and test images organized by class.
- `src/`: Contains the source code including the model, training, testing, and utility scripts.
- `requirements.txt`: List of dependencies.
- `README.md`: Project instructions and descriptions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/lithology-classification.git
cd lithology-classification

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be structured as follows:

```bash
dataset/
├── train/
├── val/
└── test/
```

## Training

To train the model:

```bash
python src/train.py
```

## Testing

To test the model:

```bash
python src/test.py
```

The model accuracy and other metrics will be printed after execution.
