# Analysis of Deep Learning Models for Emotion Recognition

This project explores the performance of various deep learning models for emotion recognition using the SEED-V dataset. The models implemented include Deep Neural Network (DNN), Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), and a hybrid CNN-LSTM model. The impact of different padding techniques (Zero Padding and Average Padding) on model performance is also evaluated.

## Dataset

The project utilizes the SEED-V dataset, which contains emotional data labeled with the following emotions:

*   Happy
*   Fear
*   Neutral
*   Sad
*   Disgust

More information about the SEED-V dataset can be found [here](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html). 

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

## Experiment Architecture

![Experiment Architecture] (src/diagram.jpeg)


## Requirements

The project requires the following Python libraries:

*numpy
*pandas
*scikit-learn
*tensorflow  
*keras       
*matplotlib
*seaborn


You can install the required libraries using pip:

* pip install -r requirements.txt

## Results

The experiments successfully demonstrated that deep learning models can achieve high accuracy in emotion recognition tasks on the SEED dataset. We observed that accuracy varies across different deep learning models based on their architecture and input preprocessing techniques.

**Key Findings:**

*   **Model Performance:** The CNN and Hybrid CNN-LSTM models significantly outperformed the DNN and LSTM models. Specifically:
    *   **DNN:** Achieved an accuracy of 68%.
    *   **LSTM:** Achieved an accuracy of 74%.
    *   **CNN:** Achieved an accuracy of 81%.
    *   **CNN-LSTM:** Achieved the highest accuracy of 83%.

*   **Padding Impact:** Both the CNN and CNN-LSTM models achieved their highest accuracy when using *average padding*. This suggests that average padding is a more effective strategy for this dataset and these architectures compared to zero padding which performed better in LSTM with an accuracy of 74%.

*   **Feature Importance:** The relatively lower performance of the LSTM model (74%) compared to the CNN and CNN-LSTM models highlights the importance of capturing local features through convolutions for this emotion recognition task. This indicates that spatial information within the input data is crucial for accurate emotion classification.

*   **Confusion Analysis:** Analysis of the confusion matrices revealed that the models had the most difficulty distinguishing between 'Fear' and 'Sad' emotions. This suggests potential similarities in the features representing these emotions in the SEED dataset and could be an area for future investigation.

*   **Detailed Results:** Further details, including specific metrics (e.g., precision, recall, F1-score) and visualizations (e.g., accuracy curves, confusion matrices), are available  within the respective model files.

## Future Work

* Explore other deep learning architectures (e.g., Transformers).
* Investigate different data augmentation techniques to improve model robustness.
* Experiment with different hyperparameter optimization methods.
* Evaluate the models on other emotion recognition datasets with other modalities (e.g., combining facial expressions and speech) to assess their generalizability.
* Explore the application of emotion recognition in real-world scenarios, such as human-computer interaction, education, or healthcare.
Address challenges related to bias, fairness, and explainability in deep learning models for emotion recognition

## Contributing
Contributions to this project are welcome. Please open an issue or submit a pull request.

