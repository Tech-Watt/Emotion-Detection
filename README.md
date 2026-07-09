# Facial Emotion Detection

A computer vision project that detects facial emotions from webcam input by extracting facial landmarks and classifying them with a machine learning model.

## Why this project stands out
This project demonstrates a full end-to-end machine learning workflow:
- collecting structured facial data,
- training a classification model,
- and deploying it in a live real-time demo.

It is a strong example of applied Python, OpenCV, and scikit-learn skills for roles in AI, computer vision, and data science.

## Key features
- Real-time face mesh landmark extraction
- Emotion data collection from webcam input
- Training pipeline for a supervised classifier
- Live demo for instant emotion prediction
- Clean, modular project structure for easier extension

## Tech stack
- Python
- OpenCV
- cvzone
- NumPy
- pandas
- scikit-learn

## Project structure
```text
Emotion-Detection/
├── data/
│   └── data.csv
├── models/
│   └── model.pkl
├── scripts/
│   ├── common.py
│   ├── generate_data.py
│   ├── train_model.py
│   └── live_demo.py
├── datagen.py
├── training.py
├── test.py
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd Emotion-Detection
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## How the project works
1. Data collection
   - Run the data collection script to capture facial landmarks and label them.
2. Model training
   - Train a classifier on the collected dataset.
3. Live prediction
   - Run the demo to classify emotions from a webcam stream.

## Usage

### Collect data
```bash
python datagen.py
```

### Train the model
```bash
python training.py
```

### Run the live demo
```bash
python test.py
```

## What I learned from building this project
- Working with computer vision pipelines
- Preparing structured data for machine learning
- Building a practical ML demo from scratch
- Organizing a project for readability and maintainability

## Future improvements
- Add more emotion classes
- Improve model accuracy with more diverse data
- Experiment with deep learning models
- Add a GUI or web interface

## License
This project is intended for educational and portfolio purposes.

