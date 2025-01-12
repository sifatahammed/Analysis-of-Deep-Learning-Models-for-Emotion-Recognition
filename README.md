# Analysis-of-Deep-Learning-Models-for-Emotion-Recognition
# Emotion Recognition using Deep Learning Models

This project explores the performance of various deep learning models for emotion recognition using the SEED-V dataset. The models implemented include Deep Neural Network (DNN), Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), and a hybrid CNN-LSTM model. The impact of different padding techniques (Zero Padding and Average Padding) on model performance is also evaluated.

## Dataset

The project utilizes the SEED-V dataset, which contains emotional data labeled with the following emotions:

*   Happy
*   Fear
*   Neutral
*   Sad
*   Disgust

More information about the SEED-V dataset can be found [here](ADD_DATASET_LINK_HERE). *It is crucial to add the correct link to the dataset documentation or download location here.* If the dataset isn't publicly available, explain how it was obtained (e.g., "obtained through research collaboration").

## Models

The following deep learning models were implemented and evaluated:

*   **DNN (Deep Neural Network):** A fully connected neural network used as a baseline model.
*   **LSTM (Long Short-Term Memory):** A recurrent neural network suitable for sequential data, capturing temporal dependencies in the emotional data.
*   **CNN (Convolutional Neural Network):** A convolutional neural network designed to extract local features from the input data.
*   **CNN-LSTM (Hybrid Model):** A combination of CNN and LSTM layers, leveraging the feature extraction capabilities of CNNs and the temporal modeling of LSTMs.

## Padding Techniques

Two padding techniques were employed to handle variable sequence lengths in the data:

*   **Zero Padding:** Padding sequences with zeros to achieve a uniform length. This can sometimes introduce bias if the padding is significant relative to the data.
*   **Average Padding:** Padding sequences with the average value of the existing data. This aims to minimize the impact of padding on the data distribution compared to zero padding.

## Project Structure
