# Fruit Classification using Convolutional Neural Networks (CNN)

## Overview

This project implements a Convolutional Neural Network (CNN) to classify images of various fruits. The goal is to build a robust image classification model capable of distinguishing between different fruit types.

The analysis and model development are performed in the `Lab_cnn_Jedata.ipynb` Jupyter Notebook.

## Methodology

The classification is achieved using a standard image processing and deep learning pipeline:

1.  **Data Acquisition and Preprocessing:**
    * Loading the dataset (likely the **Fruits-360 Dataset**).
    * Resizing images (e.g., to 50x50 pixels) and converting them into numerical arrays.
    * Encoding target labels (fruit names) for training.
2.  **Model Architecture:**
    * Building a Convolutional Neural Network (CNN) model using Keras/TensorFlow.
    * Defining convolutional layers, pooling layers, and dense layers for feature extraction and classification.
3.  **Training:**
    * Compiling the model with an appropriate loss function and optimizer.
    * Training the CNN on the prepared image data.
4.  **Evaluation and Prediction:**
    * Assessing the model's performance on a separate test set.
    * Using the trained model to make final predictions.

## Key Libraries and Technologies

* **TensorFlow / Keras:** For building and training the deep learning model.
* **OpenCV (`cv2`):** For reading and resizing image data.
* **NumPy:** For numerical operations and array manipulation.
* **Pandas:** For data handling (if applicable).
* **Matplotlib:** For data visualization.
* **Scikit-learn (`sklearn`):** For data preprocessing tasks like label encoding.

## Notebook Access

The original Jupyter Notebook can be accessed here:
[Lab\_cnn\_Jedata.ipynb](https://github.com/RachidJedata/Fruit_Classifier/blob/master/Lab_cnn_Jedata.ipynb)
