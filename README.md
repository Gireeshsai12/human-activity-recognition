# Human Activity Recognition using Smartphones and Smartwatch Sensors

## Overview
Human Activity Recognition (HAR) is a machine learning application that identifies physical activities such as walking, sitting, standing, and more using sensor data from smartphones and wearable devices.

This project builds a complete pipeline to process sensor data and classify human activities using machine learning algorithms.

## Objectives
- Analyze sensor data from smartphones and smartwatches  
- Perform data preprocessing and feature extraction  
- Train machine learning models for activity classification  
- Evaluate model performance and accuracy  

## Tech Stack
- Programming: Python  
- Libraries: NumPy, Pandas, Scikit-learn, Matplotlib  
- Tools: Jupyter Notebook / VS Code  
- Concepts: Machine Learning, Data Preprocessing, Classification  

## Dataset
The dataset consists of sensor readings collected from:
- Accelerometer  
- Gyroscope  

These signals capture motion patterns that help identify human activities.

## Project Workflow

1. Data Collection  
   Sensor data from smartphones and wearable devices  

2. Data Preprocessing  
   Cleaning missing or noisy data  
   Normalization and transformation  

3. Feature Engineering  
   Extract meaningful features from raw signals  

4. Model Training  
   Train classification models such as Logistic Regression, KNN, etc.  

5. Evaluation  
   Measure accuracy and performance metrics  

## How to Run the Project

1. Clone Repository
```bash
git clone https://github.com/Gireeshsai12/human-activity-recognition.git
cd human-activity-recognition
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the Project
```bash
python main.py
```

## Results
- The model successfully classifies human activities with good accuracy  
- Performs well on unseen test data  
- Demonstrates effectiveness of machine learning in real-world activity recognition  

## Applications
- Fitness tracking systems  
- Health monitoring  
- Smart wearable devices  
- Activity-based mobile applications  

## Project Structure
```
human-activity-recognition/
│
├── data/
├── src/
├── models/
├── notebooks/
├── main.py
├── requirements.txt
└── README.md
```

## Author
Gireesh Sai Kalluri  
Computer Science Student, University of Massachusetts Lowell
