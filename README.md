<div align="center">

# 🧠 Deep Learning for Emotion Recognition

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&duration=3000&pause=1000&center=true&vCenter=true&width=850&lines=Emotion+Recognition+using+SEED-V;DNN+%7C+LSTM+%7C+CNN+%7C+CNN-LSTM;Zero+Padding+vs+Average+Padding;Deep+Learning+%7C+EEG+Emotion+Analysis" alt="Typing SVG">

<br>

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">

<br>

<img src="https://img.shields.io/badge/SEED--V-Dataset-6366F1?style=for-the-badge" alt="SEED-V">
<img src="https://img.shields.io/badge/Task-Emotion%20Recognition-10B981?style=for-the-badge" alt="Emotion Recognition">
<img src="https://img.shields.io/badge/Best%20Accuracy-83%25-22C55E?style=for-the-badge" alt="83 percent accuracy">
<img src="https://img.shields.io/badge/License-MIT-FACC15?style=for-the-badge" alt="MIT License">

<br><br>

**A comparative study of deep learning architectures and padding strategies for emotion recognition using the SEED-V dataset.**

</div>

---

## 📌 Overview

This project investigates the use of **deep learning techniques for emotion recognition** using the **SEED-V dataset**.

The study compares four different neural-network architectures:

- 🧠 **DNN — Deep Neural Network**
- 🔄 **LSTM — Long Short-Term Memory**
- 🧩 **CNN — Convolutional Neural Network**
- 🔗 **CNN-LSTM — Hybrid CNN + LSTM**

In addition to model architecture, the project investigates the effect of different sequence-padding techniques:

- ⬛ **Zero Padding**
- 🟦 **Average Padding**

The reported experiments achieved accuracies ranging from **68% to 83%**, with the **CNN-LSTM model achieving the highest reported accuracy of 83%**.

---

# 🎯 Project Objectives

The main objectives of this research are:

1. Compare different deep learning architectures for emotion recognition.
2. Investigate the effect of padding techniques on model performance.
3. Analyze the ability of CNNs to capture local features.
4. Investigate LSTM-based temporal modeling.
5. Evaluate a hybrid CNN-LSTM architecture.
6. Analyze classification errors using confusion matrices.
7. Identify promising directions for future emotion-recognition research.

---

# 😊 Emotion Classes

The project uses five emotion categories from the SEED-V dataset:

| Emotion | Description |
|:---:|---|
| 😄 **Happy** | Positive emotional state |
| 😨 **Fear** | Fear-related emotional state |
| 😐 **Neutral** | Neutral emotional state |
| 😢 **Sad** | Sadness-related emotional state |
| 🤢 **Disgust** | Disgust-related emotional state |

---

# 📚 Dataset

This project utilizes the **SEED-V dataset** for emotion-recognition experiments.

SEED-V is an EEG-based emotion-recognition dataset developed by the Brain-Computer Interface & Machine Learning Laboratory at Shanghai Jiao Tong University.

More information about the dataset is available through the **BCMI@SJTU SEED project page**.

### Dataset Processing Pipeline

```text
                 ┌─────────────────────┐
                 │     SEED-V Dataset  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Preprocessing     │
                 │   & Segmentation    │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          Zero Padding          Average Padding
                 │                     │
                 └──────────┬──────────┘
                            │
                            ▼
                  Model Experiments
                            │
                            ▼
                       Evaluation
```
# 🏗️ Experiment Architecture

The experimental pipeline consists of three main stages:

1. **1️⃣ Data Preparation**: The SEED-V data is processed and segmented before being supplied to the neural networks.
2. **2️⃣ Model Experiments**: Multiple architectures are trained/evaluated with different padding configurations.
3. **3️⃣ Evaluation**: The trained models are evaluated and their performance metrics are collected for comparison.

---

## 🧠 Deep Learning Models

### 1. DNN — Deep Neural Network
The Deep Neural Network acts as a baseline architecture.

```text
Input
  │
  ▼
Dense Layer
  │
  ▼
Activation
  │
  ▼
Dense Layer
  │
  ▼
Output
```

DNNs use fully connected layers to learn nonlinear relationships between input features and emotion classes.

> **Reported Accuracy:** `68%`

---

### 2. LSTM — Long Short-Term Memory
LSTM is a recurrent neural-network architecture designed to learn dependencies in sequential data.

```text
Input Sequence
      │
      ▼
 ┌──────────┐
 │   LSTM   │
 └────┬─────┘
      │
      ▼
Temporal Features
      │
      ▼
Classifier
      │
      ▼
Emotion
```

LSTM is particularly useful when the ordering of observations contains meaningful information.

> **Reported Accuracy:** `74%`

---

### 3. CNN — Convolutional Neural Network
CNNs are designed to learn local patterns through convolution operations.

```text
Input
  │
  ▼
Convolution
  │
  ▼
Activation
  │
  ▼
Pooling
  │
  ▼
Feature Extraction
  │
  ▼
Classifier
  │
  ▼
Emotion
```

The reported results indicate that CNN-based feature extraction performed strongly in this experiment.

> **Reported Accuracy:** `81%`

---

### 4. CNN-LSTM — Hybrid Model
The CNN-LSTM combines convolutional feature extraction with recurrent temporal modeling.

```text
              Input
                │
                ▼
               CNN
                │
                ▼
        Local Feature Extraction
                │
                ▼
              LSTM
                │
                ▼
        Temporal Representation
                │
                ▼
            Classifier
                │
                ▼
             Emotion
```

This architecture attempts to combine the feature-learning capability of CNNs with the sequential modeling capability of LSTMs.

> **Reported Accuracy:** `83%`

---

## 📊 Summary of Model Performance

| Model Architecture | Description | Reported Accuracy |
| :--- | :--- | :---: |
| **DNN** | Baseline fully-connected network | `68%` |
| **LSTM** | Recurrent neural network for temporal data | `74%` |
| **CNN** | Local pattern extraction through convolution | `81%` |
| **CNN-LSTM** | Hybrid feature extraction & sequence modeling | **`83%`** |

---

## 🧩 Padding Techniques

A major component of this project is the comparison of two padding strategies.

### ⬛ Zero Padding
Zero padding extends a sequence by inserting zeros until all sequences reach the required length.

**Example:**
* **Original:** `[0.42, 0.61, 0.73]`
* **After Zero Padding:** `[0.42, 0.61, 0.73, 0, 0, 0]`

#### Advantages
- Simple and easy to implement
- Computationally inexpensive
- Commonly used for sequence normalization

#### Potential Limitation
If a large amount of padding is introduced, zero values may differ significantly from the original data distribution.

