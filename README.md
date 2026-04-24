## Digit Recognizer: Top 20% Solution (0.99239)

A high-performance Computer Vision pipeline built with **PyTorch** for the Kaggle Digit Recognizer (MNIST) competition. This model achieves **99.2% accuracy**, surpassing the typical 98% baseline by utilizing advanced regularization and data augmentation.

## The Architecture: "EliteCNN"
To reach super-human accuracy, the model uses a modular Convolutional Neural Network (CNN) designed for stability and generalization:
- **Convolutional Blocks:** Two double-layered stacks with `BatchNorm2d` to stabilize training and accelerate convergence.
- **Regularization:** Strategic use of `Dropout` (0.25 and 0.50) to prevent overfitting on the training set.
- **Data Augmentation:** Real-time rotation ($\pm 10^\circ$) applied during the training loop to improve robustness against varying handwriting styles.

## Performance
- **Kaggle Public Score:** 0.99239
- **Epochs:** 15
- **Hardware:** Optimized for CUDA/MPS acceleration.

## Installation & Usage
1. Clone the repo:
   ```bash
   git clone [https://github.com/ShivBoyPro/mnist-cnn-digit-recognizer.git](https://github.com/ShivBoyPro/mnist-cnn-digit-recognizer.git)