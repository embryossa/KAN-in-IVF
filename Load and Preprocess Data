import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
new_data = pd.read_excel("C:/Users/User/Desktop/IVF/GGRC/Data Analytics/NN Prediction/new_df_with_KPI.xlsx")

# Select features
selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl хор.кач-ва", "Частота оплодотворения", "Число Bl",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов",
    "KPIScore"
]

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(new_data[selected_features].values)
y = new_data['Исход переноса'].values
