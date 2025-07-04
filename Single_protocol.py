import torch
import pandas as pd
from kan import KAN
import torch.nn as nn


class KAN(nn.Module):
    def __init__(self):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(18, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = KAN()
model.load_state_dict(torch.load("C:/Users/User/Desktop/IVF/AI/DNN/Предсказания/GGRC_updated_model.pth"))
#model.load_state_dict(torch.load('C:/Users/User/Desktop/IVF/AI/DNN/Предсказания/Prediction_KAN.pth'))
# model.load_state_dict(torch.load('Prediction_KAN.pth'))

# Получение данных от пользователя
age = int(input("Введите возраст пациентки: "))
attempt_number = int(input("Введите номер попытки: "))
folicule_number = int(input("Введите количество фолликулов: "))
OCC_Number = int(input("Введите количество ооцитов на пункции: "))
fertilized = int(input("Введите количество оплодотворенных ооцитов: "))
pN = int(input("Введите количество зигот 2 pN: "))
D3 = int(input("Введите количество дробящихся эмбрионов: "))
D5 = int(input("Введите количество эмбрионов на день 5: "))
BL = int(input("Введите общее количество бластоцист: "))
GBL = int(input("Введите количество бластоцист хорошего качества: "))
ET = int(input("Введите число перенесенных эмбрионов: "))
CRYO = int(input("Введите число криоконсервированных эмбрионов: "))

# Создание нового DataFrame с введенными данными
user_input_data = {
    "Возраст": age,
    "№ попытки": attempt_number,
    "Количество фолликулов": folicule_number,
    "Число ОКК": OCC_Number,
    "Число инсеминированных": fertilized,
    "2 pN": pN,
    "Число дробящихся на 3 день": D3,
    "Число эмбрионов 5 дня": D5,
    "Число Bl": BL,
    "Число Bl хор.кач-ва": GBL,
    "Перенесено эмбрионов": ET,
    "Заморожено эмбрионов": CRYO,
}
new_df = pd.DataFrame([user_input_data])
# Расчет необходимых признаков для модели
new_df['Частота оплодотворения'] = new_df['2 pN'] / new_df['Число инсеминированных']
new_df['Частота дробления'] = new_df['Число дробящихся на 3 день'] / new_df['2 pN']
new_df['Частота формирования бластоцист'] = new_df['Число Bl'] / new_df['2 pN']
new_df['Частота формирования бластоцист хорошего качества'] = new_df['Число Bl хор.кач-ва'] / new_df['2 pN']
new_df['Частота получения ОКК'] = new_df['Число ОКК'] / new_df['Количество фолликулов']


# Создаем функцию для расчета баллов в соответствии с условиями
def calculate_kpi_score(row):
    """
    Calculates the KPI score based on the given row of data.

    Parameters:
    row (pandas.Series): A row of data containing the necessary columns for KPI calculation.

    Returns:
    int: The calculated KPI score.

    The KPI score is calculated based on specific conditions for different columns in the input row.
    The conditions are as follows:
    - If the 'Возраст' is greater than or equal to 40, add 1 to the score.
    - If the 'Возраст' is less than or equal to 36, add 5 to the score.
    - Otherwise, add 3 to the score.

    - If the 'Количество фолликулов' is greater than 15, add 5 to the score.
    - If the 'Количество фолликулов' is between 8 and 15 (inclusive), add 3 to the score.
    - Otherwise, add 1 to the score.

    - If the 'Число инсеминированных' is less than or equal to 3, add 1 to the score.
    - If the 'Число инсеминированных' is between 4 and 7 (inclusive), add 3 to the score.
    - Otherwise, add 5 to the score.

    - If the 'Частота оплодотворения' is less than 0.5, add 1 to the score.
    - If the 'Частота оплодотворения' is between 0.5 and 0.65 (inclusive), add 3 to the score.
    - Otherwise, add 5 to the score.

    - If the 'Число Bl хор.кач-ва' is equal to 0, add 1 to the score.
    - If the 'Число Bl хор.кач-ва' is less than or equal to 2, add 3 to the score.
    - Otherwise, add 5 to the score.
    """
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
new_df['KPIScore'] = new_df.apply(calculate_kpi_score, axis=1)

# Выбор признаков для прогнозирования
selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl хор.кач-ва", "Частота оплодотворения", "Число Bl",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов",
    "KPIScore",
]
# Создание нового DataFrame только с выбранными признаками
new_df_selected = new_df[selected_features]
new_data_numpy = new_df_selected.values
new_data_tensor = torch.from_numpy(new_data_numpy).float()

# Переход модели в режим оценки
model.eval()

# Получение предсказаний для новых данных
with torch.no_grad():
    predictions = model(new_data_tensor).squeeze()
    predicted_probabilities = torch.sigmoid(predictions).numpy()
    predicted_classes = (predicted_probabilities > 0.5).astype(int)

# Добавление предсказанных классов в исходный DataFrame
new_df['Исход переноса'] = predicted_classes

# Вывод предсказаний
print('Предсказанные вероятности:', predicted_probabilities)
print('Предсказанные классы:', predicted_classes)
input()
