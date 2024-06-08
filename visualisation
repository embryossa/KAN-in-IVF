import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sympy import symbols, simplify

# Загрузка данных из Excel-файла в DataFrame
df_selected = pd.read_excel("C:/Users/User/Desktop/IVF/AI/DNN/обучение и валидация/all_df_with_KPI.xlsx")

# Выбор признаков и целевой переменной
selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl хор.кач-ва", "Частота оплодотворения", "Число Bl",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов",
    "KPIScore"
]

X = df_selected[selected_features].values
y = df_selected['Исход переноса'].values

# Разделение на обучающий, валидационный и тестовый наборы
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Создание экземпляра класса StandardScaler
scaler = StandardScaler()

# Нормализация данных обучающего, валидационного и тестового наборов
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

# Преобразование данных в формат, подходящий для PyTorch
dataset = {
    'train_input': torch.from_numpy(X_train_normalized).float(),
    'val_input': torch.from_numpy(X_val_normalized).float(),
    'test_input': torch.from_numpy(X_test_normalized).float(),
    'train_label': torch.from_numpy(y_train).float(),
    'val_label': torch.from_numpy(y_val).float(),
    'test_label': torch.from_numpy(y_test).float()
}


# Инициализация модели KAN
class KAN(nn.Module):
    def __init__(self, width, grid, k):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(width[0], width[1])
        self.fc2 = nn.Linear(width[1], width[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = KAN(width=[X_train_normalized.shape[1], 10, 1], grid=5, k=3)

# Определение оптимизатора и функции потерь
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.BCEWithLogitsLoss()

# Обучение модели
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for step in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(dataset['train_input']).squeeze()
    loss = loss_fn(outputs, dataset['train_label'])
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    with torch.no_grad():
        train_preds = torch.sigmoid(outputs) > 0.5
        train_accuracy = (train_preds.long() == dataset['train_label'].long()).float().mean().item()
        train_accuracies.append(train_accuracy)

    # Оценка на валидационном наборе данных
    model.eval()
    with torch.no_grad():
        val_outputs = model(dataset['val_input']).squeeze()
        val_loss = loss_fn(val_outputs, dataset['val_label']).item()
        val_preds = torch.sigmoid(val_outputs) > 0.5
        val_accuracy = (val_preds.long() == dataset['val_label'].long()).float().mean().item()
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    if (step + 1) % 10 == 0:
        print(f'Step [{step + 1}/100], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Оценка модели на тестовых данных
model.eval()
with torch.no_grad():
    test_outputs = model(dataset['test_input']).squeeze()
    test_loss = loss_fn(test_outputs, dataset['test_label']).item()
    test_preds = torch.sigmoid(test_outputs) > 0.5
    test_accuracy = (test_preds.long() == dataset['test_label'].long()).float().mean().item()
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Визуализация процесса обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Функция для извлечения и визуализации компонентов KAN
def visualize_kan_components(model):
    # Получаем веса и смещения из обученной модели
    with torch.no_grad():
        weights_fc1 = model.fc1.weight.cpu().numpy()
        biases_fc1 = model.fc1.bias.cpu().numpy()
        weights_fc2 = model.fc2.weight.cpu().numpy()
        biases_fc2 = model.fc2.bias.cpu().numpy()

    # Построение графиков весов и смещений
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(weights_fc1[0], label='Weights of fc1')
    plt.plot(biases_fc1, label='Biases of fc1')
    plt.legend()
    plt.title('fc1 weights and biases')

    plt.subplot(1, 2, 2)
    plt.plot(weights_fc2[0], label='Weights of fc2')
    plt.plot(biases_fc2, label='Biases of fc2')
    plt.legend()
    plt.title('fc2 weights and biases')

    plt.show()

    return weights_fc1, biases_fc1, weights_fc2, biases_fc2


# Визуализация и получение весов и смещений
weights_fc1, biases_fc1, weights_fc2, biases_fc2 = visualize_kan_components(model)


# Функция для выполнения символьной регрессии
def symbolic_regression(weights, biases):
    x = symbols('x')
    func = 0
    for w, b in zip(weights, biases):
        func += w * x + b
    return simplify(func)


# Пример для одного из слоев
symbolic_func_fc1 = symbolic_regression(weights_fc1[0], biases_fc1)
symbolic_func_fc2 = symbolic_regression(weights_fc2[0], biases_fc2)

print(f"Symbolic Function for fc1: {symbolic_func_fc1}")
print(f"Symbolic Function for fc2: {symbolic_func_fc2}")
