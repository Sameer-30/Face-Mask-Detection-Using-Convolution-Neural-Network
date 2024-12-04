# Face Mask Detection using Convolutional Neural Networks (CNN)

## Overview
This project implements a **Face Mask Detection model** using **Convolutional Neural Networks (CNNs)** to classify images of people wearing face masks and those not wearing face masks. This model aims to help in monitoring public health measures and ensuring the safety of individuals by leveraging computer vision techniques.

The project utilizes a pre-collected dataset of images categorized into two classes: `With Mask` and `Without Mask`. The goal is to train a CNN to accurately classify the images into these categories.

## Project Workflow

1. **Dataset Collection**  
   The project uses a public dataset (or you can use your custom dataset) containing images of individuals with and without face masks. The dataset is divided into two categories:  
   - **With Mask**  
   - **Without Mask**  

2. **Data Preprocessing**  
   The images are resized, normalized, and augmented to ensure a diverse and balanced dataset for model training.

3. **Model Building**  
   The **Convolutional Neural Network (CNN)** model is built using multiple convolutional layers, pooling layers, and fully connected layers. The CNN is well-suited for image classification tasks like this.

4. **Model Training**  
   The model is trained on the preprocessed dataset using backpropagation and gradient descent. The loss function and optimizer are tuned to maximize classification accuracy.

5. **Model Evaluation**  
   After training, the model is evaluated on a test set to measure performance metrics like **accuracy**, **precision**, and **recall**. The results are visualized with confusion matrices and other plots.

6. **Final Prediction**  
   The trained model is used to predict whether a person in a given image is wearing a mask or not. The results are displayed as class labels or with bounding boxes on the images.

## Tools & Libraries Used
- **Python**: The programming language used for the implementation.
- **TensorFlow/Keras**: Libraries used for building and training the CNN model.
- **OpenCV**: For image processing and handling.
- **Matplotlib**: For visualizing training performance and results.
- **NumPy**: For numerical operations and data manipulation.
- **Pandas**: For data management (optional depending on dataset format).

## Results
The CNN model demonstrates high accuracy in detecting whether a person is wearing a face mask or not. The model's performance is evaluated based on various metrics, and it shows how well it generalizes to new, unseen data.

