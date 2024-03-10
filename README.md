# Fashion-MNIST-Classifier

This repository contains a machine learning project focused on image classification using the Fashion MNIST dataset. Various algorithms will be explored to analyze and improve classification performance.

## Fashion MNIST Dataset

The Fashion MNIST dataset is a collection of 28x28 grayscale images representing 10 different fashion product categories. Each category is represented by 7,000 images, making a total of 70,000 images in the dataset. The pixel values range from 0 to 255. You can find out more about the dataset by visting Keras' [website](https://keras.io/api/datasets/fashion_mnist/).

## Project Overview

The goal of this project is to build and evaluate machine learning models for image classification on the Fashion MNIST dataset. The primary focus has been on using the k-Nearest Neighbors (KNN) algorithm, complemented by preprocessing techniques such as Min-Max scaling and dimensionality reduction using Principal Component Analysis (PCA).

## K-Nearest Neighbors (KNN) Results

### Data Preprocessing

Before applying the KNN algorithm, the dataset was preprocessed by reshaping the images from 3D arrays to 2D arrays and flattening them into 784-dimensional vectors.

### Hyperparameter Tuning

The optimal value for the number of neighbors (k) in KNN was determined through experimentation, with values ranging from [1, 3, 7, 12, 20, 30, 50, 75, 100]. The performance of each configuration was evaluated on a validation set, and the optimal k value was selected based on the highest validation accuracy.

### Model Evaluation

The final KNN model was trained using the optimal k value determined during hyperparameter tuning. Its performance was assessed on a separate test set, and the results, including accuracy and a confusion matrix, are provided below.

<!-- Include KNN results here -->

## Future Work

While KNN has been the focus thus far, future iterations of this project may explore other machine learning algorithms, including deep learning approaches, to further enhance image classification performance.

<!-- Include space for future algorithm results -->

Feel free to contribute or provide feedback on this project!
