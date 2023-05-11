# HEART_DISEASE_BINARY_CLASSIFICATION

This repository contains code for a binary classification model that predicts the likelihood of heart disease based on several medical attributes.

The dataset used for training and testing the model is the "Heart Disease UCI" dataset, which is publicly available on Kaggle and UCI Machine Learning Repository. The dataset contains 13 attributes such as age, sex, chest pain type, resting blood pressure, serum cholestoral, fasting blood sugar, etc. and the target variable is a binary value indicating the presence or absence of heart disease.

Model Architecture
The model is implemented using PyTorch and consists of 3 fully connected layers with ReLU activation and a final sigmoid activation function. The architecture is as follows:

Layer 1: Linear(in_features=13, out_features=26)
Activation: ReLU

Layer 2: Linear(in_features=26, out_features=20)
Activation: ReLU

Layer 3: Linear(in_features=20, out_features=1)
Activation: Sigmoid
Training
The dataset is split into a training set (67% of the data) and a test set (33% of the data) using the train_test_split function from scikit-learn. The data is normalized using the StandardScaler from scikit-learn.

The model is trained using binary cross-entropy loss (BCELoss) and the Adam optimizer with a learning rate of 0.001. The best model weights are saved using early stopping based on test loss. If the test loss does not improve for three consecutive epochs, training stops early.

Usage
The model weights can be saved using torch.save(model.state_dict(),'HeartDiseasePredictionModel.pth'). To load the model, create an instance of the HeartDiseasePredictionModel class and load the saved weights using model.load_state_dict(torch.load('HeartDiseasePredictionModel.pth')).

To make predictions, call the forward method of the HeartDiseasePredictionModel class on a tensor of inputs. The output will be a tensor of predicted probabilities, which can be rounded to 0 or 1 for binary classification.

Requirements
The code requires the following packages:

pandas
scikit-learn
PyTorch
These packages can be installed using pip:

pip install pandas scikit-learn torch

Author
Divine Jojolola
