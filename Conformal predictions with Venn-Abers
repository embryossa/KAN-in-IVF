import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from keras.models import load_model
from venn_abers import VennAbersCalibrator
import pickle

# Define the KAN model class
class KAN(nn.Module):
    def __init__(self):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(18, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the KAN model
kan_model = KAN()
kan_model.load_state_dict(torch.load("C:/Users/User/PycharmProjects/pythonProject/KAN/GGRC_updated_model.pth"))

# Load the DNN model
dnn_model = load_model("C:/Users/User/Desktop/IVF/AI/DNN/FINAL/neural_network_model.h5")

# Load data from Excel file into DataFrame
new_df = pd.read_excel("C:/Users/User/Desktop/IVF/AI/DNN/обучение и валидация/all_df_with_KPI.xlsx")

# Preprocess the data
new_df.fillna(0, inplace=True)
selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl", "Число Bl хор.кач-ва", "Частота оплодотворения",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов",
    "KPIScore"
]

X = new_df[selected_features].values
y = new_df['Исход переноса'].values

# Scaling and normalizing the data
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_scaled)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Custom wrapper for KAN model to work with Venn-Abers
class KANWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            # Output is the probability for the positive class (class 1)
            output = torch.sigmoid(self.model(X_tensor)).numpy().flatten()
            # Calculate the probability for the negative class (class 0) as 1 - output
            return np.vstack([1 - output, output]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def fit(self, X, y=None):
        # Dummy fit method to conform to sklearn's API
        return self

    def get_params(self, deep=True):
        # Return parameters for sklearn compatibility
        return {"model": self.model}

    def set_params(self, **params):
        # Set parameters for sklearn compatibility
        for key, value in params.items():
            setattr(self, key, value)
        return self


# Custom wrapper for DNN model to work with Venn-Abers
class DNNWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        # Output is the probability for the positive class (class 1)
        output = self.model.predict(X).flatten()
        # Calculate the probability for the negative class (class 0) as 1 - output
        return np.vstack([1 - output, output]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def fit(self, X, y=None):
        # Dummy fit method to conform to sklearn's API
        return self

    def get_params(self, deep=True):
        # Return parameters for sklearn compatibility
        return {"model": self.model}

    def set_params(self, **params):
        # Set parameters for sklearn compatibility
        for key, value in params.items():
            setattr(self, key, value)
        return self

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

# Function to convert probabilities to binary predictions with a given threshold
def proba_to_binary(proba, threshold=0.5):
    return (proba >= threshold).astype(int)

# Create wrappers for KAN and DNN models
kan_wrapper = KANWrapper(kan_model)
dnn_wrapper = DNNWrapper(dnn_model)

# Initialize Venn-Abers calibrator for both models
kan_va_calibrator = VennAbersCalibrator(estimator=kan_wrapper, inductive=True, cal_size=0.2, random_state=101)
dnn_va_calibrator = VennAbersCalibrator(estimator=dnn_wrapper, inductive=True, cal_size=0.2, random_state=101)

# Fit the calibrator on the training set
kan_va_calibrator.fit(X_train, y_train)
dnn_va_calibrator.fit(X_train, y_train)

# Generate probabilities and class predictions for KAN model on the test set
kan_proba = kan_va_calibrator.predict_proba(X_test)
kan_proba_flat = kan_proba[:, 1]  # Assuming the positive class is at index 1
kan_pred = proba_to_binary(kan_proba_flat)

# Generate probabilities and class predictions for DNN model on the test set
dnn_proba = dnn_va_calibrator.predict_proba(X_test)
dnn_proba_flat = dnn_proba[:, 1]  # Assuming the positive class is at index 1
dnn_pred = proba_to_binary(dnn_proba_flat)

# Check shapes
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of kan_proba_flat: {kan_proba_flat.shape}")
print(f"Shape of dnn_proba_flat: {dnn_proba_flat.shape}")

# Ensure that y_test, kan_pred, and dnn_pred have the same number of samples
if y_test.shape[0] != kan_pred.shape[0]:
    raise ValueError(f"Mismatch between y_test and kan_pred shapes: {y_test.shape[0]} != {kan_pred.shape[0]}")
if y_test.shape[0] != dnn_pred.shape[0]:
    raise ValueError(f"Mismatch between y_test and dnn_pred shapes: {y_test.shape[0]} != {dnn_pred.shape[0]}")

# Evaluate KAN model performance after calibration
kan_accuracy = accuracy_score(y_test, kan_pred)
kan_precision = precision_score(y_test, kan_pred)
kan_recall = recall_score(y_test, kan_pred)
kan_f1 = f1_score(y_test, kan_pred)
kan_roc_auc = roc_auc_score(y_test, kan_proba_flat)  # Use probabilities directly
kan_mcc = matthews_corrcoef(y_test, kan_pred)

# Evaluate DNN model performance after calibration
dnn_accuracy = accuracy_score(y_test, dnn_pred)
dnn_precision = precision_score(y_test, dnn_pred)
dnn_recall = recall_score(y_test, dnn_pred)
dnn_f1 = f1_score(y_test, dnn_pred)
dnn_roc_auc = roc_auc_score(y_test, dnn_proba_flat)  # Use probabilities directly
dnn_mcc = matthews_corrcoef(y_test, dnn_pred)

# Print performance metrics
print("KAN Model Performance after Venn-Abers Calibration:")
print(f"Accuracy: {kan_accuracy:.4f}, Precision: {kan_precision:.4f}, Recall: {kan_recall:.4f}, F1 Score: {kan_f1:.4f}, AUC: {kan_roc_auc:.4f}, MCC: {kan_mcc:.4f}")

print("DNN Model Performance after Venn-Abers Calibration:")
print(f"Accuracy: {dnn_accuracy:.4f}, Precision: {dnn_precision:.4f}, Recall: {dnn_recall:.4f}, F1 Score: {dnn_f1:.4f}, AUC: {dnn_roc_auc:.4f}, MCC: {dnn_mcc:.4f}")

from sklearn.metrics import brier_score_loss

kan_brier_score = brier_score_loss(y_test, kan_proba_flat)
dnn_brier_score = brier_score_loss(y_test, dnn_proba_flat)

print(f"KAN Brier Score: {kan_brier_score:.4f}")
print(f"DNN Brier Score: {dnn_brier_score:.4f}")

from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    ece = np.abs(prob_true - prob_pred).mean()
    return ece

kan_ece = expected_calibration_error(y_test, kan_proba_flat)
dnn_ece = expected_calibration_error(y_test, dnn_proba_flat)

print(f"KAN ECE: {kan_ece:.4f}")
print(f"DNN ECE: {dnn_ece:.4f}")

def maximum_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    mce = np.max(np.abs(prob_true - prob_pred))
    return mce

kan_mce = maximum_calibration_error(y_test, kan_proba_flat)
dnn_mce = maximum_calibration_error(y_test, dnn_proba_flat)

print(f"KAN MCE: {kan_mce:.4f}")
print(f"DNN MCE: {dnn_mce:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import scipy.stats as stats

# Define the function to plot calibration curves
def calibration_plot(y_true, y_prob, model_name, label_prefix=''):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} {label_prefix}')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()

# Normalize predictions to [0, 1]
def normalize_probabilities(proba):
    return (proba - np.min(proba)) / (np.max(proba) - np.min(proba))

# Pre-calibration predictions for KAN
kan_proba_raw = kan_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()
kan_proba_raw = normalize_probabilities(kan_proba_raw)  # Normalize

# Pre-calibration predictions for DNN
dnn_proba_raw = dnn_model.predict(X_test)  # Ensure this is a (n_samples, 1) shape if it's not already
dnn_proba_raw = dnn_proba_raw.flatten()  # Flatten if necessary

plt.figure(figsize=(14, 10))

# Plot pre-calibration calibration
calibration_plot(y_test, kan_proba_raw, 'KAN (Raw)', label_prefix='Before Calibr.: ')
calibration_plot(y_test, dnn_proba_raw, 'DNN (Raw)', label_prefix='Before Calibr.: ')

# Apply Venn-Abers calibration
kan_proba_cal = kan_va_calibrator.predict_proba(X_test)[:, 1]
dnn_proba_cal = dnn_va_calibrator.predict_proba(X_test)[:, 1]

# Plot post-calibration calibration
calibration_plot(y_test, kan_proba_cal, 'KAN (Calibrated)', label_prefix='After Calibr.: ')
calibration_plot(y_test, dnn_proba_cal, 'DNN (Calibrated)', label_prefix='After Calibr.: ')

# Plot perfect calibration line
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot Before and After Venn-Abers Calibration')
plt.legend()
plt.show()
