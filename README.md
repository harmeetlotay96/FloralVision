# Flower Image Classification with CNN and Data Augmentation
This project demonstrates the use of Convolutional Neural Networks (CNNs) combined with data augmentation techniques to classify images of flowers into five distinct categories using TensorFlow and Keras.

## Project Overview
The objective is to build a robust image classifier capable of distinguishing between different species of flowers. By employing data augmentation, the model generalizes better to unseen data, enhancing its predictive performance.

## Dataset
The model is trained on a dataset comprising 3,670 images categorized into five classes:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

The dataset is publicly available and can be downloaded from the TensorFlow datasets repository.

## Key Components
- Data Loading: Utilizes tf.keras.utils.image_dataset_from_directory to efficiently load and preprocess images.
- Data Augmentation: Applies random transformations such as flipping, rotation, and zooming to increase dataset variability and reduce overfitting.
- Model Architecture: Constructs a CNN with multiple convolutional and pooling layers, followed by dense layers for classification.
- Training: Compiles the model using the Adam optimizer and trains it with a categorical cross-entropy loss function.
- Evaluation: Assesses model performance on a validation set and visualizes training history.

## Getting Started
Prerequisites
-Python 3.x
-TensorFlow
-Keras
-Matplotlib
-NumPy
Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/harmeetlotay96/tf_keras_cnn_flowers_classification.git
Navigate to the project directory:

bash
Copy
Edit
cd tf_keras_cnn_flowers_classification
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Ensure the dataset is available in the specified directory or modify the data loading path accordingly.

Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook flowers_image_augmentation_classification.ipynb
Follow the notebook cells to train and evaluate the model.

Results
The model achieves an accuracy of approximately X% on the validation set, demonstrating effective classification capabilities enhanced by data augmentation techniques.
