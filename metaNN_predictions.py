import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from keras.models import load_model
import pickle

# Define the KAN model
class KAN(nn.Module):
    def __init__(self):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(18, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

kan_model = KAN()
kan_model.load_state_dict(torch.load("C:/Users/User/PycharmProjects/pythonProject/KAN/GGRC_updated_model.pth"))

# Загрузка модели DNN
dnn_model = load_model("C:/Users/User/PycharmProjects/pythonProject/DNN for GGRC/GGRC_tuned_model.h5")

# Загрузка метамодели
with open("meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

# Load data from Excel file into DataFrame
new_df = pd.read_excel("C:/Users/User/Desktop/IVF/AI/DNN/обучение и валидация/all_df_with_KPI.xlsx")

# Replace missing values with 0 in the DataFrame
new_df.fillna(0, inplace=True)

# Selected features for prediction
selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl", "Число Bl хор.кач-ва", "Частота оплодотворения",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов",
    "KPIScore"
]

# Replace infinities with NaN and then replace NaN with 0
new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
new_df.fillna(0, inplace=True)

# Scaling data before normalization
min_max_scaler = MinMaxScaler()
new_df_scaled = min_max_scaler.fit_transform(new_df[selected_features])

# Normalizing data
scaler = StandardScaler()
new_df_normalized = scaler.fit_transform(new_df_scaled)

# Converting to tensor for KAN model
new_data_tensor = torch.from_numpy(new_df_normalized).float()

# Getting predictions from the KAN model
kan_model.eval()
with torch.no_grad():
    kan_predictions = torch.sigmoid(kan_model(new_data_tensor)).numpy()

# Reshaping data for the DNN model
new_data_reshaped = new_df_normalized.reshape((new_df_normalized.shape[0], new_df_normalized.shape[1], 1))

# Getting predictions from the DNN model
dnn_predictions = dnn_model.predict(new_data_reshaped).ravel()

# Creating meta model predictions
stacked_features = np.column_stack((kan_predictions, dnn_predictions))
meta_predictions = meta_model.predict_proba(stacked_features)[:, 1]
meta_predictions_labels = (meta_predictions > 0.5).astype(int)

# Adding predicted classes and probabilities to the original DataFrame
new_df['Исход переноса КАН'] = (kan_predictions > 0.5).astype(int)
new_df['Исход переноса DNN'] = (dnn_predictions > 0.5).astype(int)  # Changed threshold to 0.5 for consistency
new_df['Исход переноса метамодель'] = meta_predictions_labels
new_df['Вероятность положительного исхода метамодель'] = meta_predictions

# Save DataFrame to Excel file
new_df.to_excel('предсказания_метамодель.xlsx', index=False)

# Calculating and printing frequencies
pregnancy_yes_count = new_df['Исход переноса метамодель'].sum()
total_count = len(new_df['Исход переноса метамодель'])
pregnancy_frequency = pregnancy_yes_count / total_count
print(f'ЧНБ_прогноз: {pregnancy_frequency:.2%}')

pregnancy_real_count = new_df['Исход переноса'].sum()
real_pregnancy_frequency = pregnancy_real_count / total_count
print(f'ЧНБ_актуальный: {real_pregnancy_frequency:.2%}')
