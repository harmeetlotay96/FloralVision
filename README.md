# 🌸 FloralVision - Flower Image Classification with CNN & Data Augmentation

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify images of flowers into five categories. It incorporates **data augmentation** to enhance model generalization and improve classification accuracy.

## 📌 Project Overview

The goal is to develop a **deep learning model** that can accurately classify flower species while leveraging data augmentation techniques to prevent overfitting.

### 🌿 Dataset

The dataset consists of **3,670 images** of flowers categorized into five classes:
- 🌼 Daisy  
- 🌾 Dandelion  
- 🌹 Roses  
- 🌻 Sunflowers  
- 🌷 Tulips  

The dataset is publicly available via **TensorFlow datasets**.

### 🔑 Key Features

- **Data Loading:** Uses `tf.keras.utils.image_dataset_from_directory` for efficient dataset handling.
- **Data Augmentation:** Applies flipping, rotation, and zooming to improve training robustness.
- **CNN Architecture:** Builds a convolutional neural network with multiple layers for feature extraction.
- **Training & Optimization:** Utilizes **Adam optimizer** and **categorical cross-entropy loss**.
- **Evaluation & Visualization:** Assesses performance using a validation set and plots accuracy/loss trends.

---

## 🚀 Getting Started

### 🔧 Prerequisites

Ensure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy

### 📥 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/harmeetlotay96/tf_keras_cnn_flowers_classification.git
   ```
