import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
df = pd.read_excel("C:/Users/User/PycharmProjects/pythonProject/KAN/предсказания_метамодель.xlsx")

# List of numeric columns to use for the correlation matrix
numeric_columns = ["Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
                   "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
                   "Число Bl хор.кач-ва", "Частота оплодотворения", "Число Bl",
                   "Частота дробления", "Частота формирования бластоцист",
                   "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
                   "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов",
                   "KPIScore", "Вероятность метамодель", "Вероятность KAN", "Вероятность DNN", "Вероятность EVO"]

# Select only the numeric columns from the DataFrame
numeric_df = df[numeric_columns]

# Summary statistics for specific columns
specific_columns = ["KPIScore", "Вероятность метамодель", "Вероятность KAN", "Вероятность DNN", "Вероятность EVO"]
summary_stats = df[specific_columns].describe()
print("Summary Statistics:\n", summary_stats)

# Подготовка данных для одного общего графика
df_melted = numeric_df.melt(id_vars='KPIScore',
                            value_vars=['Вероятность метамодель', 'Вероятность KAN', 'Вероятность DNN', 'Вероятность EVO'],
                            var_name='Model', value_name='Probability')

# Создание scatter plot с разными цветами и стилями для каждого набора данных
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_melted, x='KPIScore', y='Probability', hue='Model', style='Model', palette='viridis', s=100,
                alpha=0.7)

# Настройка заголовков и меток осей
plt.title('KPIScore vs Probability for Different Models', fontsize=16)
plt.xlabel('KPIScore', fontsize=14)
plt.ylabel('Probability', fontsize=14)

# Добавление сетки и настройка спайнов
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.legend(title='Model')
plt.show()

# Compute the correlation matrix for the numeric columns
correlation_matrix = numeric_df.corr()

# Параметры стиля и шрифтов
plt.figure(figsize=(14, 8))
plt.rcParams.update({'font.size': 6})
# Построение тепловой карты корреляции
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=10)
plt.xlabel('Признаки')
plt.ylabel('Признаки')
plt.show()
# Model Prediction Comparison
# Assuming 'Исход переноса' contains the true labels
true_labels = df['Исход переноса']

# Confusion matrix and classification report for each model
models = {
    "KAN": df['Исход переноса КАН'],
    "DNN": df['Исход переноса DNN'],
    "MetaModel": df['Исход переноса метамодель'],
    "EVO": df['Исход переноса EVO']
}

for model_name, predictions in models.items():
    print(f"Confusion Matrix for {model_name} Model:\n", confusion_matrix(true_labels, predictions))
    print(f"Classification Report for {model_name} Model:\n", classification_report(true_labels, predictions))

# Создание фигуры для объединенной гистограммы
plt.figure(figsize=(12, 8))

# Гистограмма для Вероятность метамодель
sns.histplot(df['Вероятность метамодель'], kde=True, color='#9370DB', label='MetaModel', alpha=0.5, bins=30)

# Гистограмма для Вероятность KAN
sns.histplot(df['Вероятность KAN'], kde=True, color='#4169E1', label='KAN', alpha=0.5, bins=30)

# Гистограмма для Вероятность DNN
sns.histplot(df['Вероятность DNN'], kde=True, color='#66CDAA', label='DNN', alpha=0.5, bins=30)

# Гистограмма для Вероятность EVO
sns.histplot(df['Вероятность EVO'], kde=True, color='#F5F5DC', label='EVO', alpha=0.5, bins=30)

# Настройка заголовка и меток осей
plt.title('Distribution of Predicted Probabilities for Different Models', fontsize=16)
plt.xlabel('Predicted Probability', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Добавление легенды
plt.legend(title='Model')
plt.tight_layout()
plt.show()


# Параметры стиля и шрифтов
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 10})  # Установка размера шрифта для всего графика
plt.grid(True, linestyle='--', alpha=0.6)  # Добавление сетки

# Создание единого scatter plot для всех моделей
sns.scatterplot(x='Вероятность метамодель', y='Частота формирования бластоцист хорошего качества', data=df, s=100,
                alpha=0.7, color='#9370DB', label='MetaModel')
sns.scatterplot(x='Вероятность KAN', y='Частота формирования бластоцист хорошего качества', data=df, s=100, alpha=0.7,
                color='#4169E1', label='KAN')
sns.scatterplot(x='Вероятность DNN', y='Частота формирования бластоцист хорошего качества', data=df, s=100, alpha=0.7,
                color='#66CDAA', label='DNN')
sns.scatterplot(x='Вероятность EVO', y='Частота формирования бластоцист хорошего качества', data=df, s=100, alpha=0.7,
               color='#FF9966', label='EVO')
# Настройка заголовка и меток осей
plt.title('Вероятность vs. Частота формирования бластоцист хорошего качества', fontsize=16)
plt.xlabel('Вероятность', fontsize=14)
plt.ylabel('Частота формирования бластоцист хорошего качества', fontsize=14)

# Настройка легенды
plt.legend(title='Модель')

# Настройка сетки
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()  # Для улучшения расположения элементов графика
plt.show()

# Создание фигуры с subplot'ами
plt.figure(figsize=(20, 5))

# Боксплот 1: KPIScore vs. Вероятность MetaModel
plt.subplot(141)
sns.boxplot(x='KPIScore', y='Вероятность метамодель', data=df)
plt.title('KPIScore vs. Predicted MetaModel Probability')
plt.xlabel('KPIScore')
plt.ylabel('Вероятность метамодель')

# Боксплот 2: KPIScore vs. Вероятность KAN
plt.subplot(142)
sns.boxplot(x='KPIScore', y='Вероятность KAN', data=df)
plt.title('KPIScore vs KAN Probability')
plt.xlabel('KPIScore')
plt.ylabel('Вероятность KAN')

# Боксплот 3: KPIScore vs. Вероятность DNN
plt.subplot(143)
sns.boxplot(x='KPIScore', y='Вероятность DNN', data=df)
plt.title('KPIScore vs DNN Probability')
plt.xlabel('KPIScore')
plt.ylabel('Вероятность DNN')

# Боксплот 4: KPIScore vs. Вероятность EVO
plt.subplot(144)
sns.boxplot(x='KPIScore', y='Вероятность EVO', data=df)
plt.title('KPIScore vs EVO Probability')
plt.xlabel('KPIScore')
plt.ylabel('Вероятность EVO')

plt.tight_layout()
plt.show()
