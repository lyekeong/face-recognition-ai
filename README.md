# Face Recognition AI

A comprehensive face recognition project implementing multiple AI techniques and algorithms to detect, classify, and recognize faces in images and video streams.

## 📋 Project Overview

This project is an assignment that explores various artificial intelligence and computer vision techniques for face recognition. It implements and compares different approaches including:

- **Convolutional Neural Networks (CNN)** for face classification
- **YOLO (You Only Look Once)** for real-time face detection
- **Machine Learning methods** for comparison and evaluation
- **GUI interface** for user-friendly interaction


## 🎯 Key Features

- **Multi-method Implementation**: Compare performance of CNN, YOLO, and ML-based approaches
- **Face Detection**: Real-time face detection using YOLO
- **Face Classification**: Classify detected faces using trained CNN models
- **GUI Application**: User-friendly interface for testing and demonstration
- **Performance Comparison**: Comprehensive evaluation using metrics like accuracy, precision, recall, and F1-score
- **Dataset Support**: Preprocessing and augmentation pipelines for various datasets

## 📁 Project Structure

```
face-recognition-ai/
├── cnn/                          # CNN-based face classification
│   ├── cnn.py                   # CNN model implementation
│   ├── models/                  # Trained CNN models
│   └── outputs_cnn/             # Model outputs and results
│
├── yolo/                        # YOLO-based face detection
│   ├── train_classifier.py     # Training script for YOLO
│   ├── evaluate.py             # Evaluation and testing script
│   ├── yolov8n-face.pt         # Pre-trained YOLO model
│   └── output/                 # Detection results and outputs
│
├── machine learning/            # Machine Learning implementations
│   └── src/
│       └── ml.py               # ML-based classification methods
│
├── gui/                        # Graphical User Interface
│   └── gui.py                  # GUI application for testing
│
├── comparison/                 # Model comparison and analysis
│   └── compare.py              # Comparison metrics and visualization
│
├── dataset_split/              # Dataset preprocessing
│   └── Train/test split utilities
│
└── README.md                   # Current file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: (See requirements below)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lyekeong/face-recognition-ai.git
cd face-recognition-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: torch==2.9.1+cu126 & torchvision==0.24.1+cu126 need to download independently if failed by using command ```pip install -r requirements.txt```

Common dependencies include:
- `opencv-python` - Image processing and face detection
- `tensorflow` or `pytorch` - Deep learning frameworks
- `scikit-learn` - Machine learning algorithms
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `tkinter` - GUI development (usually included with Python)

### Usage

#### CNN-based Face Classification
```bash
python cnn/cnn.py
```

#### YOLO Face Detection and Evaluation
```bash
# Train the classifier
python yolo/train_classifier.py

# Evaluate performance
python yolo/evaluate.py
```

#### Machine Learning Approach
```bash
python machine\ learning/src/ml.py
```

#### Compare Results
```bash
python comparison/compare.py
```

#### Launch GUI Application
```bash
python gui/gui.py
```

## 🔍 Methodology

### Data Preprocessing
- Image resizing and normalization
- Data augmentation for improved model robustness
- Train/test split for proper evaluation
- Feature extraction and encoding

### Model Training
- Each group member implements different algorithms
- Hyperparameter tuning for optimal performance
- Cross-validation for reliability

### Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positive predictions
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 📊 Results and Comparison

The project includes comprehensive comparisons between different methods:
- Performance analysis across all three approaches
- Advantages and disadvantages of each method
- Recommendations for specific use cases
- Real-time demonstration capabilities

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Application of CNN for deep learning-based face recognition
- ✅ Real-time object detection using YOLO
- ✅ Traditional machine learning techniques for classification
- ✅ Data preprocessing and feature engineering
- ✅ Model evaluation and comparison
- ✅ Software development with Python
- ✅ GUI development for user interaction

## 📝 Requirements

See `requirements.txt` for a complete list of Python dependencies. Key packages include:
- OpenCV (cv2)
- TensorFlow/PyTorch
- scikit-learn
- NumPy
- Matplotlib
- Tkinter (for GUI)


## 🔗 References

- UCI Machine Learning Repository: https://archive.ics.uci.edu/datasets
- Kaggle Datasets: https://www.kaggle.com/datasets
- YOLO Official: https://docs.ultralytics.com/
- TensorFlow Documentation: https://www.tensorflow.org/
- OpenCV Documentation: https://docs.opencv.org/

---

**Note**: This project is developed as part of an Artificial Intelligence course assignment. All code is original work by the group members and not derived from others' work.
