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
git clone https://github.com/sheharyaar/lithology-identification-core-images
cd lithology-identification-core-images
```
2. Extract the dataset

```bash
tar -xzvf dataset.tar.gz
```

3. Install dependencies in a virtuallenv:

```bash
python -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

## Dataset Structure

The dataset should be structured as follows:

```bash
dataset/
├── train/
├── val/
└── test/
```
The model accuracy and other metrics will be printed after execution.

