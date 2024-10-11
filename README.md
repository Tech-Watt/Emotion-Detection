Facial Emotion Detection using Face Mesh and Machine Learning
Project Overview
This project involves creating a facial emotion detection system using face mesh points captured via a webcam or video, combined with a machine learning model. The project generates face data (x, y coordinates of facial landmarks) using OpenCV and cvzone, stores it in a CSV file, trains a machine learning model to classify emotions, and uses the trained model to predict emotions from live video feed.

The project consists of three major components:

Data Generation: Captures face mesh landmarks and stores them with emotion labels.
Model Training: Trains a machine learning model (Logistic Regression) on the face mesh data.
Emotion Prediction: Uses the trained model to classify the emotion in real-time based on face mesh data.

Table of Contents
Installation
Project Structure
How It Works
Usage
Customization
Dependencies
License
Installation

Clone the Repository
Open your terminal and run:
git clone <repository_url>
cd <repository_directory>
Install Dependencies
Make sure you have Python installed. Install the required packages using:
pip install -r requirements.txt
Project Structure

├── data.csv                   # CSV file for storing face mesh data with emotion labels
├── datagen.py                 # Script to generate and save face data
├── model.pkl                  # Pre-trained machine learning model (Logistic Regression)
├── requirements.txt           # List of project dependencies
├── test.py                    # Script to predict emotions using the trained model
└── training.py                # Script for training the model on face data

How It Works
1. Data Generation (datagen.py)
This script uses a webcam or video input to capture face mesh points (468 points per frame) using cvzone.FaceMeshModule.
It flattens these points and saves them in a CSV file (data.csv) along with the emotion label (e.g., 'happy', 'sad').

The key facial landmarks (x, y coordinates) are extracted for each frame.
The emotion label is inserted manually into the script and saved with the data.
2. Model Training (training.py)
This script loads the captured face mesh data from data.csv and trains a Logistic Regression model on the data to classify emotions. It uses a pipeline with standard scaling and logistic regression. The trained model is then saved as model.pkl.

The dataset is split into features (landmark points) and labels (emotion classes).
A pipeline is created with StandardScaler and LogisticRegression.
The trained model is evaluated and saved for later use.
3. Emotion Prediction (test.py)
This script uses the trained model (model.pkl) to predict emotions in real-time from the webcam feed. It captures face mesh points in each frame, feeds them into the model, and displays the predicted emotion on the video.

Real-time prediction using the trained machine learning model.
The live video feed is processed frame by frame to detect face mesh points.
The predicted emotion is overlaid on the video.
Usage
Step 1: Data Generation
Run the datagen.py script to capture and store face mesh data. The data is stored in data.csv with the class label for each frame.


python datagen.py
Step 2: Model Training
Train the model using the captured data by running the training.py script. The trained model will be saved as model.pkl.


python training.py
Step 3: Emotion Prediction
Run the test.py script to start real-time emotion prediction using your webcam.
python test.py

Customization
Adding New Emotions: To train the model on additional emotions, modify the class_name in datagen.py to the desired emotion and capture new data.
Model Parameters: You can replace the Logistic Regression model with other classifiers (e.g., SVM or Random Forest) by modifying the training.py script.
Video Input: Change the cap = cv2.VideoCapture(1) in datagen.py and test.py to cap = cv2.VideoCapture('video.mp4') if you want to use a video file instead of live webcam input.

Dependencies
cvzone: For face mesh detection and easy text overlay.
OpenCV: For video processing and capturing frames from the webcam.
NumPy: For handling numerical operations.
Scikit-learn: For machine learning algorithms and data preprocessing.
To install the dependencies, simply run:
pip install -r requirements.txt

License
This project is licensed under the MIT License. You are free to use, modify, and distribute this project.

