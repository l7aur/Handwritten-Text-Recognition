# Handwritten Character Recognition (A-Z + MNIST) using ResNet

This project trains a deep learning model (ResNet-based) to recognize handwritten digits (0–9) and letters (A–Z).  
It combines the **MNIST dataset** and a custom **A–Z handwriting dataset**, pre-processes the data, trains a ResNet model, evaluates results, and visualizes predictions.

## Features

- Custom ResNet implementation using Keras/TensorFlow  
- Training on MNIST + A–Z dataset  
- Data augmentation via `ImageDataGenerator`  
- Polynomial learning rate decay  
- Class weighting for imbalanced datasets  
- Automatic training plot generation  
- Prediction montage visualization  
- Model saved as `.keras` file  

## Requirements

Install required packages:

```bash
pip install tensorflow keras scikit-learn imutils opencv-python matplotlib numpy
```

## Project structure

    project/
    │
    ├── core/
    │   ├── models.py       # ResNet implementation
    │   ├── az_dataset.py   # Dataset loaders
    │
    ├── train.py            # Main training script (the code you provided)
    ├── README.md

## Datasets

### MNIST

Loaded automatically using the helper function.

### A–Z Handwriting Dataset

You must download it separately.
Pass its folder path to the --az argument.

## Usage

```bash
python train.py \
    --az path/to/az_dataset \
    --model output/handwriting_model \
    --plot output/training_plot.png
```

### Arguments

| Argument  | Description                                   |
| --------- | --------------------------------------------- |
| `--az`    | Path to A–Z dataset directory                 |
| `--model` | Output path (model will be saved as `.keras`) |
| `--plot`  | Output training history plot (PNG)            |

## Training configuration

- Epochs: 75
- Batch size: 128
- Initial LR: 0.01
- Optimizer: SGD + Momentum
- Learning rate schedule: Polynomial decay
- Image size: 32×32 grayscale

## Bibliographical research

### Residual Networks (ResNets)

A **Residual Network (ResNet)** is a deep convolutional neural network architecture introduced by He et al. in the paper **“Deep Residual Learning for Image Recognition” (2015)**.  
ResNet solves a major problem in deep learning: **degradation in very deep networks**.

Normally, as the number of layers increases, accuracy gets worse because gradients become too small (vanishing gradients).  
ResNet addresses this by introducing **skip connections** (also called shortcuts).

#### Skip Connections
A skip connection lets the input of a layer bypass several transformations and be added directly to the output: 

`Output = F(x) + x`


This forces the network to learn a **residual** (the difference from the identity).  
As a result:

- gradients flow more easily,
- very deep networks can be trained,
- accuracy improves significantly.

ResNet models come in many variants (ResNet-18, 34, 50, 101), and your project uses a **custom lightweight ResNet** suitable for 32×32 handwritten characters.

### 2. Bottleneck Residual Modules

My project uses **bottleneck-style residual blocks**, designed to reduce computation and increase efficiency.

Each block uses three convolutions:

1. **1×1 convolution (reduce channels)**  
2. **3×3 convolution (process features)**  
3. **1×1 convolution (expand channels back to full size)**  

This structure reduces the computational cost while maintaining representational power.

The custom ResNet is defined inside `core/models.py` and built as follows:

- Input: 32×32 grayscale image
- Initial convolution: 3×3
- Residual stages: Controlled by the stages and filters parameters:
- Stages define how many residual blocks per level
- Filters define the number of feature maps
- Batch Normalization + ReLU after each convolution
- Average pooling near the end
- Fully connected layer for classification with softmax

The project uses an `ImageDataGenerator` to help the model generalize:
- rotation
- zoom
- width/height shifts
- shearing
- nearest fill mode

These operations simulate real-world handwriting variations.
