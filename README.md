# 🧠 Image Classification with ANN  
*A Deep Learning Assignment – Petra Christian University*

---

## 📚 Academic Context

This project was created as part of the **Deep Learning** course assignment at **Petra Christian University**. It demonstrates how to build, train, and evaluate an Artificial Neural Network (ANN) for image classification tasks using Python and TensorFlow/Keras.

---

## 🎯 Objective

To gain hands-on experience with core concepts in deep learning — such as neural network architecture, activation functions, loss functions, and performance evaluation — by implementing a simple yet functional image classification model using an ANN.

---

## 🛠️ Technologies Used

- **Language**: Python  
- **Environment**: Jupyter Notebook  
- **Framework**: TensorFlow / Keras  
- **Libraries**:
  - `numpy`
  - `matplotlib`
  - `tensorflow` / `keras`
  - `sklearn` (for evaluation)

---

## 🚀 How It Works

1. **Dataset Loading**:
   - Load a dataset (Fashion MNIST) using TensorFlow Datasets.

2. **Data Preprocessing**:
   - Normalize image data (e.g., pixel values between 0 and 1)
   - Flatten image input (if needed) for feeding into ANN

3. **Model Architecture**:
   - Build a Sequential model using Keras
   - Layers typically include:
     - Dense layers with ReLU activation
     - Dropout for regularization
     - Output layer with Softmax activation for multi-class classification

4. **Training**:
   - Compile model with appropriate loss function (`categorical_crossentropy` or `sparse_categorical_crossentropy`)
   - Fit model to training data
   - Track accuracy and loss over epochs

5. **Evaluation**:
   - Test the model on validation or test set
   - Plot confusion matrix and accuracy metrics

6. **Visualization**:
   - Plot training/validation accuracy and loss over time
   - Display sample predictions

---

## 📁 File Overview

- `Image_Classification_with_ANN.ipynb` – Main Jupyter notebook with code for loading data, preprocessing, building the ANN model, training, evaluating, and visualizing results.

---

## ✅ Requirements

Install dependencies:

```bash
pip install tensorflow matplotlib scikit-learn
