# Pneumonia Detection Using Deep Learning

## Project Overview

Pneumonia is a serious respiratory disease that can lead to severe complications if not diagnosed early. This project presents an AI-powered diagnostic system that automatically classifies chest X-ray images as either Normal or Pneumonia.

Two deep learning approaches were implemented and compared:

1. Custom CNN trained from scratch
2. ResNet-50 Transfer Learning Model

The final system was deployed as a Streamlit web application that allows healthcare professionals to upload chest X-rays and receive real-time AI-assisted predictions.

---

## Dataset

Dataset Source:
Kaggle Chest X-Ray Images (Pneumonia)

Total Images: 5,856

| Category  | Images |
| --------- | ------ |
| Normal    | 1,583  |
| Pneumonia | 4,273  |

Image Size:
224 × 224 pixels

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Streamlit
* PIL (Pillow)

---

## Models Implemented

### CNN From Scratch

Architecture:

* Conv2D
* Batch Normalization
* ReLU
* Max Pooling
* Dropout
* Fully Connected Layers

Training:

* Epochs: 20
* Optimizer: Adam
* Learning Rate: 0.001

---

### ResNet-50 Transfer Learning

Fine-Tuning Strategy:

* Pretrained on ImageNet
* Frozen feature extractor
* Unfrozen Layer4
* Custom Classification Head

Training:

* Epochs: 15
* Differential Learning Rates
* Adam Optimizer

---

## Performance Comparison

| Metric           | CNN    | ResNet-50 |
| ---------------- | ------ | --------- |
| Accuracy         | 83.49% | 83.65%    |
| ROC-AUC          | 0.9140 | 0.9632    |
| Pneumonia Recall | 96%    | 99%       |
| False Negatives  | 16     | 2         |

Best Model:
ResNet-50 Transfer Learning

---

## Key Achievements

* Trained on 5,856 chest X-rays
* Achieved ROC-AUC of 0.9632
* Achieved 99% Pneumonia Recall
* Reduced False Negatives by 87.5%
* Real-time Streamlit Deployment
* Medical AI Application

---

## Streamlit Application Features

* Upload Chest X-ray
* Real-Time Prediction
* Confidence Scores
* Model Selection
* Medical Disclaimer
* Interactive Dashboard

---

## Results

The ResNet-50 model correctly identified:

* 388 out of 390 pneumonia cases
* Missed only 2 positive patients

This makes the model highly suitable for AI-assisted clinical screening.

---

## Future Improvements

* Grad-CAM Visualization
* Multi-Class Pneumonia Classification
* EfficientNet Benchmarking
* Cloud Deployment
* Explainable AI Integration

---

## Author

Abdullah Nazir

Computer Science Student

Interests:

* Artificial Intelligence
* Medical AI
* Deep Learning
* Computer Vision
* Healthcare Analytics

---

## License

MIT License
