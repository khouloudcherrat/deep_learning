# Deep Learning Labs
<p align="center">
 <a href="https://numpy.org/"><img src="https://numpy.org/images/logos/numpy_logo.svg" alt="NumPy" height="70"></a>
  <a href="https://pandas.pydata.org/"><img src="https://pandas.pydata.org/docs/_static/pandas.svg" alt="pandas" height="70"></a>
  <a href="https://www.tensorflow.org/"><img src="https://www.tensorflow.org/images/tf_logo_social.png" alt="TensorFlow" height="70"></a>
  <a href="https://matplotlib.org/"><img src="https://matplotlib.org/stable/_static/logo2.svg" alt="Matplotlib" height="70"></a>
  <a href="https://scikit-learn.org/"><img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="scikit-learn" height="70"></a>
</p>

This repository contains a series of deep learning labs developed during my master's degree in computer science. These labs provide practical experience with various deep learning concepts using frameworks such as TensorFlow and Keras.

## Table of Contents

- [Lab 1: Fit the Digit Dataset with Keras](#lab-1-fit-the-digit-dataset-with-keras)
  - [Description](#description)
  - [Steps](#steps)
  - [Results](#results)
  - [Requirements](#requirements-1)

- [Lab 2: Predicting Product Ratings from Reviews](#lab-2-predicting-product-ratings-from-reviews)
  - [Overview](#overview-1)
  - [Dataset](#dataset-1)
  - [Text Preprocessing](#text-preprocessing)
  - [Model Training](#model-training)
  - [Performance Evaluation](#performance-evaluation)
  - [Usage](#usage-1)
  - [Requirements](#requirements-2)

- [Lab 3: Self-Supervised Learning](#lab-3-self-supervised-learning)
  - [Overview](#overview-2)
  - [Objectives](#objectives)
  - [Technologies Used](#technologies-used)
  - [Dataset](#dataset-2)

- [Lab 4: LSTM Model for Time Series Forecasting](#lab-4-lstm-model-for-time-series-forecasting)
  - [Overview](#overview-3)
  - [Dataset](#dataset-3)
  - [Requirements](#requirements-3)

- [Lab 5: Image Classification with Convolutional Neural Networks](#lab-5-image-classification-with-convolutional-neural-networks)
  - [Overview](#overview-4)
  - [Requirements](#requirements-4)

## Lab 1: Fit the Digit Dataset with Keras

### Description
In this lab, we utilize the digits dataset from `sklearn` to train a neural network model using Keras. The objective is to classify handwritten digits (0-9) with high accuracy.

### Steps
1. **Load the Dataset**: The digits dataset is loaded using `sklearn.datasets`.
2. **Plot the Dataset**: A visualization of the digits is created to understand the data better.
3. **Split the Dataset**: The dataset is divided into training and testing sets.
4. **One-Hot Encode Labels**: The target labels are transformed into a one-hot encoded format.
5. **Build the Keras Model**: A neural network model with four hidden layers is constructed.
6. **Compile the Model**: The model is compiled with categorical cross-entropy loss and the Adam optimizer.
7. **Train the Model**: The model is trained with a validation split to monitor performance.
8. **Evaluate the Model**: The accuracy of the model is evaluated on the test set.

### Results
The model achieved approximately 92.8% accuracy on the test dataset.

### Requirements
- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- matplotlib

## Lab 2: Predicting Product Ratings from Reviews

### Overview
This lab focuses on predicting product ratings based on customer reviews using a portion of the **Amazon_Unlocked_Mobile.csv** dataset from Kaggle. The primary goal is to analyze the reviews and predict the corresponding ratings using machine learning techniques.

### Dataset
The dataset contains the following columns:
- **Product Name**
- **Brand Name**
- **Price**
- **Rating** (target variable)
- **Reviews** (input feature)
- **Review Votes**

For this lab, we focus on 'Reviews' as the input (X) and 'Rating' as the output (y).

### Text Preprocessing
To work with text data, we utilize TensorFlow's `TextVectorization` layer to convert the reviews into a document-term matrix, which quantifies the frequency of terms in the reviews.

### Model Training
We build a Multi-Layer Perceptron (MLP) model to learn from the processed text data. The lab addresses potential challenges, such as the vanishing gradient problem, and explores various solutions, including:
- Adjusting the learning rate
- Using sparsity-promoting activation functions (e.g., ReLU)
- Implementing normalization techniques (e.g., Batch Normalization)

### Performance Evaluation
The model's performance is evaluated using learning curves, classification reports, and confusion matrices to assess accuracy and generalization capabilities.

### Usage
To run the lab:
1. Clone this repository.
2. Ensure you have the required packages installed (e.g., TensorFlow, scikit-learn, etc.).
3. Execute the Jupyter notebook provided in this repository.

### Requirements
- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib
- pandas

## Lab 3: Self-Supervised Learning

### Overview
This lab explores self-supervised learning through the implementation of various autoencoders using the MNIST dataset. The primary objectives include building different types of autoencoders, evaluating their performance in classification tasks, and testing clustering methods on encoded data.

### Objectives
- Construct different types of autoencoders:
  - Simple Autoencoder
  - Stacked Autoencoder
  - Variational Autoencoder
- Implement clustering techniques using K-Means:
  - Clustering on the original dataset
  - Clustering on encoded data
  - Clustering after projecting data into latent space using UMAP
- Evaluate the autoencoders through image reconstruction and morphing techniques.
- Apply deep clustering methods to improve clustering results.

### Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV

### Dataset
The lab utilizes the MNIST dataset, which consists of handwritten digits, to train and test the autoencoders.

## Lab 4: LSTM Model for Time Series Forecasting

### Overview
This project implements a Long Short-Term Memory (LSTM) model to forecast monthly precipitation data using time series analysis.

### Dataset
The dataset used in this project can be found [here](http://www.i3s.unice.fr/~riveill/dataset/precipitation.csv.zip). It contains monthly precipitation data recorded over several years.

### Requirements
To run this project, you'll need the following packages:
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `scikit-learn`

## Lab 5: Image Classification with Convolutional Neural Networks

### Overview
This lab utilizes convolutional neural networks (CNNs) for image classification. We explore the use of pre-trained models, specifically VGG16, and implement techniques such as transfer learning, fine-tuning, and visualization of model features.

### Requirements
Make sure you have the following libraries installed:
- Python >= 3.7
- TensorFlow >= 2.0
- Keras
- NumPy
- Matplotlib
- scikit-learn
- tf-keras-vis
- 
## Acknowledgements
- Dataset sourced from Kaggle.
- TensorFlow documentation for guidance on text preprocessing and model building.
