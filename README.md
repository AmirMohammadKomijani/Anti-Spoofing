# Face Anti-Spoofing Project using Frequency and LBP Domains

## Project Overview
This project addresses the problem of face spoofing detection by leveraging both deep learning and feature extraction techniques in the frequency and Local Binary Pattern (LBP) domains. The goal is to accurately differentiate between genuine and spoofed faces using a combination of machine learning models and specialized feature extraction methods.

### Key Objectives
- Develop a robust anti-spoofing system capable of detecting fake faces in images.
- Explore both direct image-based deep learning approaches and feature-based methods (LBP and frequency domain).
- Evaluate and compare the performance of various CNN architectures.

## Datasets
Two well-known datasets for face anti-spoofing were used:
1. **CASIA-FASD**: Contains videos of genuine and spoofed faces, providing diverse spoofing types like printed photo and video replay.
2. **CelebA-Spoof**: A large-scale face anti-spoofing dataset containing annotated images of spoofed faces, including various spoof types and lighting conditions.

These datasets provide a rich variety of spoofing scenarios, enabling the models to generalize well across different spoofing techniques.

## Models Used
Several models were employed to tackle the anti-spoofing task, covering both simple and complex architectures:
1. **Simple CNN**: A basic convolutional neural network for quick testing and baseline comparison.
2. **Bigger-complex CNN**: A more complex CNN architecture with additional layers for better feature extraction and improved accuracy.
3. **MobileNet**: A lightweight, efficient architecture suitable for mobile and embedded systems, balancing accuracy and computational cost.
4. **InceptionV3**: A powerful model that captures multi-scale features using its inception modules, improving robustness against varied spoofing methods.

## Approach and Domains
The project employs two primary approaches for anti-spoofing:

### 1. **Deep Learning on Images**
   - In this approach, models are trained directly on raw image data from the datasets.
   - Each model processes images to learn distinguishing features between real and spoofed faces.

### 2. **Feature Extraction Techniques**
   - **Local Binary Pattern (LBP)**: This method extracts texture features by encoding pixel-wise intensity changes, which are effective for detecting spoofing based on surface texture differences.
   - **Frequency Domain Analysis**: By transforming images to the frequency domain, the models can detect subtle variations that are often present in spoofed images but not in genuine ones.

## Installation and Setup
To set up the environment, ensure you have Python 3.8 or above. Install the required libraries:
```bash
pip install tensorflow keras opencv-python scikit-learn
