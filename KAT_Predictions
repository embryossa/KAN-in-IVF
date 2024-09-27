import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
from kan import *
from mambular.models import FTTransformerClassifier

# Загрузка обученной модели
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


class EnsembleModel(nn.Module):
    def __init__(self, kan_model, ft_model, feature_names):
        super(EnsembleModel, self).__init__()
        self.kan_model = kan_model
        self.ft_model = ft_model
        self.kan_weight = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.ft_weight = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.feature_names = feature_names

    def forward(self, x):
        kan_output = self.kan_model(x)
        kan_output = torch.sigmoid(kan_output)

        # Преобразуем входные данные в DataFrame для FTTransformer
        x_df = pd.DataFrame(x.cpu().numpy(), columns=self.feature_names)

        # Получаем вероятности для FTTransformer без индексирования
        ft_output = torch.tensor(self.ft_model.predict_proba(x_df)).float().to(x.device)

        # Убедимся, что размерности совпадают
        kan_output = kan_output.squeeze()
        ft_output = ft_output.squeeze()

        ensemble_output = self.kan_weight * kan_output + self.ft_weight * ft_output
        return ensemble_output.unsqueeze(1)  # Возвращаем тензор размерности (batch_size, 1)

kan_model = KAN()
kan_model.load_state_dict(torch.load("C:/Users/User/PycharmProjects/pythonProject/KAN/GGRC_updated_model.pth"))

# Загрузка сохранённой модели FTTransformerClassifier
ft_model = FTTransformerClassifier(
    d_model=64,
    n_layers=8,
    numerical_preprocessing="ple",
    n_bins=50
)

# Предположим, что модель была сохранена как ft_transformer.pth
ft_model=load("C:/Users/User/Desktop/IVF/AI/DNN/FINAL/FTTransformer.joblib")

# Загружаем веса ансамблевой модели
ensemble_model = EnsembleModel(kan_model, ft_model, feature_names=[
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl", "Число Bl хор.кач-ва", "Частота оплодотворения",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества",
    "Частота получения ОКК", "Число эмбрионов 5 дня",
    "Заморожено эмбрионов", "Перенесено эмбрионов", "KPIScore"
])
ensemble_model.load_state_dict(torch.load('KAT.pth'))

# Загрузка новых данных
new_data = pd.read_excel("C:/Users/User/PycharmProjects/pythonProject/DNN for GGRC/DraftIII.xlsx")
# Replace missing values with 0 in the DataFrame
new_data.fillna(0, inplace=True)
# Преобразование столбцов в числовой формат, если они не содержат числовые значения
columns_to_convert = ['День переноса']

for col in columns_to_convert:
    new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

new_data['Частота оплодотворения'] = new_data['2 pN'] / new_data['Число инсеминированных']
new_data['Частота дробления'] = new_data['Число дробящихся на 3 день'] / new_data['2 pN']
new_data['Частота формирования бластоцист'] = new_data['Число Bl'] / new_data['2 pN']
new_data['Частота формирования бластоцист хорошего качества'] = new_data['Число Bl хор.кач-ва'] / new_data['2 pN']
new_data['Частота получения ОКК'] = new_data['Число ОКК'] / new_data['Количество фолликулов']
new_data['Исход переноса'] = new_data['Исход переноса'].apply(lambda x: 1 if 'Yes' in x else 0)

# Создаем функцию для расчета баллов в соответствии с вашими условиями
def calculate_kpi_score(row):
    kpi_score = 0

    # Условия для столбца "Возраст"
    if row['Возраст'] >= 40:
        kpi_score += 1
    elif row['Возраст'] <= 36:
        kpi_score += 5
    else:
        kpi_score += 3

    # Условия для столбца "Количество фолликулов"
    if row['Количество фолликулов'] > 15:
        kpi_score += 5
    elif 8 <= row['Количество фолликулов'] <= 15:
        kpi_score += 3
    else:
        kpi_score += 1

    # Условия для столбца "Число инсеминированных"
    if row['Число инсеминированных'] <= 3:
        kpi_score += 1
    elif 4 <= row['Число инсеминированных'] <= 7:
        kpi_score += 3
    else:
        kpi_score += 5

    # Условия для столбца "Частота оплодотворения"
    if row['Частота оплодотворения'] < 0.5:
        kpi_score += 1
    elif row['Частота оплодотворения'] <= 0.65:
        kpi_score += 3
    else:
        kpi_score += 5

    # Условия для столбца "Число Bl хор.кач-ва"
    if row['Число Bl хор.кач-ва'] == 0:
        kpi_score += 1
    elif row['Число Bl хор.кач-ва'] <= 2:
        kpi_score += 3
    else:
        kpi_score += 5

    return kpi_score

# Применяем функцию для создания столбца "KPIScore"
new_data['KPIScore'] = new_data.apply(calculate_kpi_score, axis=1)

# Нормализация новых данных (предполагается, что те же признаки, что и на этапе обучения)
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
new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
new_data.fillna(0, inplace=True)

# Выбираем только нужные признаки для предсказания
X_new = new_data[selected_features].values

# Предполагается, что используем тот же StandardScaler, что и на обучении
scaler = StandardScaler()
X_new_normalized = scaler.fit_transform(X_new)

# Конвертация данных в тензоры для подачи в модель
new_input = torch.FloatTensor(X_new_normalized)

# Предобработка данных
new_data_selected = new_data[selected_features]

# Получение предсказаний от ансамблевой модели
with torch.no_grad():
    new_input = torch.tensor(new_data_selected.values).float()
    predictions = ensemble_model(new_input)

# Получение вероятностей предсказаний для положительного класса
probabilities = predictions.squeeze().numpy()

# Добавление предсказанных классов и вероятностей к исходной таблице
new_data['Исход переноса KAT'] = (probabilities > 0.45).astype(int)
new_data['Probability KAT'] = probabilities

# Сохранение DataFrame в Excel файл
new_data.to_excel('предсказания_KAT.xlsx', index=False)

# Подсчет и вывод частоты беременности
pregnancy_yes_count = new_data['Исход переноса KAT'].sum()
total_count = len(new_data['Исход переноса KAT'])
pregnancy_frequency = pregnancy_yes_count / total_count
print(f'ЧНБ_прогноз: {pregnancy_frequency:.2%}')

pregnancy_real_count = new_data['Исход переноса'].sum()  # Предполагается, что у вас есть этот столбец в new_data
real_pregnancy_frequency = pregnancy_real_count / total_count
print(f'ЧНБ_актуальный: {real_pregnancy_frequency:.2%}')
