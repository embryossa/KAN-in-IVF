import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from keras.models import load_model
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
dnn_model = load_model("C:/Users/User/PycharmProjects/pythonProject/DNN for GGRC/GGRC_tuned_model.h5")

# Load the metamodel
with open("meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

# Load data from Excel file into DataFrame
new_df = pd.read_excel("C:/Users/User/Desktop/IVF/GGRC/Data Analytics/NN Prediction/new_df_with_KPI.xlsx")

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

# Initialize KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
roc_auc_list = []
mcc_list = []

for train_index, test_index in kf.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Convert data to tensor for KAN model
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    # Get KAN model predictions
    kan_model.eval()
    with torch.no_grad():
        kan_predictions = torch.sigmoid(kan_model(X_test_tensor)).numpy()

    # Reshape data for DNN model
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Get DNN model predictions
    dnn_predictions = dnn_model.predict(X_test_reshaped).ravel()

    # Create meta model predictions
    stacked_features = np.column_stack((kan_predictions, dnn_predictions))
    meta_predictions = meta_model.predict_proba(stacked_features)[:, 1]
    meta_predictions_labels = (meta_predictions > 0.5).astype(int)

    # Calculate metrics
    accuracy_list.append(accuracy_score(y_test, meta_predictions_labels))
    precision_list.append(precision_score(y_test, meta_predictions_labels))
    recall_list.append(recall_score(y_test, meta_predictions_labels))
    f1_list.append(f1_score(y_test, meta_predictions_labels))
    roc_auc_list.append(roc_auc_score(y_test, meta_predictions))
    mcc_list.append(matthews_corrcoef(y_test, meta_predictions_labels))

# Calculate mean and standard deviation for each metric
accuracy_mean, accuracy_sd = np.mean(accuracy_list), np.std(accuracy_list)
precision_mean, precision_sd = np.mean(precision_list), np.std(precision_list)
recall_mean, recall_sd = np.mean(recall_list), np.std(recall_list)
f1_mean, f1_sd = np.mean(f1_list), np.std(f1_list)
roc_auc_mean, roc_auc_sd = np.mean(roc_auc_list), np.std(roc_auc_list)
mcc_mean, mcc_sd = np.mean(mcc_list), np.std(mcc_list)

# Print the average metrics with standard deviations
print(f'Average Accuracy: {accuracy_mean:.2f} ± {accuracy_sd:.2f}')
print(f'Average Precision: {precision_mean:.2f} ± {precision_sd:.2f}')
print(f'Average Recall: {recall_mean:.2f} ± {recall_sd:.2f}')
print(f'Average F1 Score: {f1_mean:.2f} ± {f1_sd:.2f}')
print(f'Average ROC-AUC: {roc_auc_mean:.2f} ± {roc_auc_sd:.2f}')
print(f'Average MCC: {mcc_mean:.2f} ± {mcc_sd:.2f}')
