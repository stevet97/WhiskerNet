# WhiskerNet
A Convolutional Neural Network (CNN) built with Keras and TensorFlow to classify cat vs dog images, achieving 81% validation accuracy using dropout regularization and hyperparameter tuning.

 CNN for Cat vs Dog Image Classification with Dropout Regularization

## Project Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify images of cats and dogs with high accuracy. The goal was to experiment with dropout regularization and different model architectures to reduce overfitting and achieve robust generalization. The project is part of my AI learning journey, using Keras and TensorFlow.

## Problem Solved
Image classification between cats and dogs can be a challenging task for machines due to subtle differences. This CNN model is designed to distinguish between these two categories, achieving a validation accuracy of over 81% while minimizing overfitting through the use of dropout layers.

## Key Features
- **Convolutional Neural Network (CNN)**: Utilized convolutional layers to extract important image features and classify them as cat or dog.
- **Dropout Regularization**: Integrated dropout layers to prevent overfitting, improving the model's ability to generalize to unseen data.
- **Experimentation**: Explored different dropout rates and experimented with adjusting batch size, learning rates, and early stopping to optimize performance.
- **Final Accuracy**: The model achieved a training accuracy of 80.64% and validation accuracy of 81.35% with a stable validation loss of 0.4007.

## Experimentation and Findings
During the course of model development, I experimented with several aspects:
- **Dropout Layers**: Dropout was applied after the dense layers to reduce overfitting, particularly when training on a relatively small dataset.
- **Batch Size and Learning Rates**: I adjusted batch sizes and learning rates to optimize model convergence and prevent stagnation during training.
- **Convolutional Layers**: Explored different kernel sizes and pooling strategies to maximize the model's ability to distinguish features.
- **Early Stopping**: Considered implementing early stopping after validation accuracy plateaued to avoid unnecessary training.

## Model Architecture
The CNN architecture consisted of:
- **Convolutional Layers**: Multiple convolutional layers followed by max-pooling to extract features from the input images.
- **Dense Layers**: Fully connected layers for classification, followed by a softmax output layer.
- **Dropout**: Applied after the dense layers to improve generalization.
- **Optimizer**: Adam optimizer with learning rate tuning.

## Results
After 30 epochs of training, the model achieved:
- **Training Accuracy**: 80.64%
- **Validation Accuracy**: 81.35%
- **Validation Loss**: 0.4007

While the model stabilized, further experimentation with hyperparameters and data augmentation could potentially improve performance.

## Future Improvements
- **Data Augmentation**: Introduce techniques like flipping, rotation, and zoom to increase the diversity of training data and improve generalization.
- **Transfer Learning**: Consider applying transfer learning using pre-trained models (e.g., VGG16 or ResNet) to leverage pre-learned features for image classification.
- **Hyperparameter Tuning**: Further optimization of batch sizes, learning rates, and dropout rates to boost accuracy.
- **Experiment with Deeper Architectures**: Add more convolutional layers to increase the model's capacity to learn complex features.

## Finding the Dataset on Kaggle

## Dataset
The dataset used for training the CNN model is available on Kaggle:
- [Cat and Dog Image Dataset](https://www.kaggle.com/datasets/stethomas/cat-and-dog-image-dataset-for-whiskernet/data)

To download the dataset programmatically:
```python
!pip install kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_files('your-username/cat-and-dog-image-dataset', path='data/', unzip=True)
