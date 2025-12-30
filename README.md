Handwritten Character Recognition & Disease Prediction

This project contains two machine learning applications:
Handwritten Digit Recognition using a Convolutional Neural Network (CNN) trained on the MNIST dataset.
Disease Prediction System using Logistic Regression based on user-entered symptoms.
Both projects demonstrate basic machine learning and deep learning concepts using Python.

📌 Project 1: Handwritten Character Recognition
    📖 Description
This module recognizes handwritten digits (0–9) using a CNN model built with TensorFlow and Keras. The model is trained on the MNIST dataset and can predict digits from test images.
🛠️ Technologies Used
Python
TensorFlow / Keras
NumPy
Matplotlib
📊 Dataset
MNIST dataset (60,000 training images, 10,000 test images)
Image size: 28 × 28 grayscale
    ⚙️ Steps Performed
Load and normalize the MNIST dataset
Build a CNN model
Train the model for 5 epochs
Evaluate accuracy on test data
Predict and display a sample digit
    ▶️ Output
Displays test accuracy
Shows a handwritten digit image
Prints predicted and actual label

📌 Project 2: Disease Prediction System
    📖 Description
This module predicts a disease based on symptoms entered by the user. It uses Logistic Regression trained on a CSV dataset.
🛠️ Technologies Used
Python
Pandas
Scikit-learn
📊 Dataset
CSV file (disease.csv)
Features:
Fever
Cough
Headache
Fatigue
Target:
Disease
    ⚙️ Steps Performed
Load dataset from CSV
Split data into training and testing sets
Train Logistic Regression model
Evaluate accuracy
Take user input for symptoms
Predict disease based on input
    ▶️ User Input Format

1 = Yes
0 = No
Example:

Fever: 1
Cough: 0
Headache: 1
Fatigue: 1

    📦 Installation & Requirements
Install required libraries using:

pip install tensorflow numpy matplotlib pandas scikit-learn

    ▶️ How to Run
Make sure Python is installed
Ensure disease.csv is in the same folder
Run the script:

python main.py

📁 Project Structure

project-folder/
│── handwritten_recognition.py
│── disease_prediction.py
│── disease.csv
│── README.md

    🎯 Learning Outcomes
Understanding CNNs for image classification
Applying Logistic Regression for classification
Data preprocessing and model evaluation
Taking real-time user input for predictions

  👤 Author
(Rohit singh)
