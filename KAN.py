import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    'train_label': torch.from_numpy(y_train).float(),  # Для BCEWithLogitsLoss метки должны быть float
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


model = KAN(width=[X_train_normalized.shape[1], 10, 1], grid=5,
            k=3)  # Для бинарной классификации выходной слой имеет размер 1

# Определение оптимизатора и функции потерь
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.BCEWithLogitsLoss()

# Списки для сохранения метрик
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_precisions = []
train_recalls = []
train_f1s = []
val_precisions = []
val_recalls = []
val_f1s = []
test_precisions = []
test_recalls = []
test_f1s = []

# Обучение модели
for step in range(100):
    model.train()
    optimizer.zero_grad()

    outputs = model(dataset['train_input']).squeeze()
    loss = loss_fn(outputs, dataset['train_label'])
    loss.backward()
    optimizer.step()

    # Оценка на обучающем наборе данных
    train_losses.append(loss.item())
    train_preds = torch.sigmoid(outputs) > 0.5
    train_accuracy = (train_preds.long() == dataset['train_label'].long()).float().mean().item()
    train_accuracies.append(train_accuracy)
    train_precision = precision_score(dataset['train_label'].cpu(), train_preds.long().cpu())
    train_recall = recall_score(dataset['train_label'].cpu(), train_preds.long().cpu())
    train_f1 = f1_score(dataset['train_label'].cpu(), train_preds.long().cpu())
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)

    # Оценка на валидационном наборе данных
    model.eval()
    with torch.no_grad():
        val_outputs = model(dataset['val_input']).squeeze()
        val_loss = loss_fn(val_outputs, dataset['val_label']).item()
        val_losses.append(val_loss)
        val_preds = torch.sigmoid(val_outputs) > 0.5
        val_accuracy = (val_preds.long() == dataset['val_label'].long()).float().mean().item()
        val_accuracies.append(val_accuracy)
        val_precision = precision_score(dataset['val_label'].cpu(), val_preds.long().cpu())
        val_recall = recall_score(dataset['val_label'].cpu(), val_preds.long().cpu())
        val_f1 = f1_score(dataset['val_label'].cpu(), val_preds.long().cpu())
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

    if (step + 1) % 1 == 0:
        print(
            f'Step [{step + 1}/20], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

# Оценка модели на тестовых данных
model.eval()
with torch.no_grad():
    test_outputs = model(dataset['test_input']).squeeze()
    test_loss = loss_fn(test_outputs, dataset['test_label']).item()
    test_preds = torch.sigmoid(test_outputs) > 0.5
    test_accuracy = (test_preds.long() == dataset['test_label'].long()).float().mean().item()
    test_precision = precision_score(dataset['test_label'].cpu(), test_preds.long().cpu())
    test_recall = recall_score(dataset['test_label'].cpu(), test_preds.long().cpu())
    test_f1 = f1_score(dataset['test_label'].cpu(), test_preds.long().cpu())

print(f'Точность модели на тестовых данных: {test_accuracy:.4f}')
print(f'Precision на тестовом наборе данных: {test_precision:.4f}')
print(f'Recall на тестовом наборе данных: {test_recall:.4f}')
print(f'F1-score на тестовом наборе данных: {test_f1:.4f}')

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


# Построение ROC-AUC и PRC графиков
def plot_roc_prc(y_true, y_scores, dataset_type):
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset_type}')
    plt.legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prc_auc = auc(recall, precision)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_type}')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


# Валидация
model.eval()
with torch.no_grad():
    val_outputs = model(dataset['val_input']).squeeze()
    val_scores = torch.sigmoid(val_outputs).cpu().numpy()
    plot_roc_prc(dataset['val_label'].cpu().numpy(), val_scores, 'Validation')

# Тестирование
model.eval()
with torch.no_grad():
    test_outputs = model(dataset['test_input']).squeeze()
    test_scores = torch.sigmoid(test_outputs).cpu().numpy()
    plot_roc_prc(dataset['test_label'].cpu().numpy(), test_scores, 'Test')

# Сохранение модели в файл
torch.save(model.state_dict(), 'Prediction_KAN.pth')

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

X_val_tensor = dataset['val_input']
y_val_tensor = dataset['val_label']
# Получаем некалиброванные выходы модели
with torch.no_grad():
    outputs = model(X_val_tensor)
    uncalibrated_probs = torch.sigmoid(outputs).cpu().numpy()

# Применяем изотоническую регрессию для калибровки
ir = IsotonicRegression(out_of_bounds='clip')
calibrated_probs = ir.fit_transform(uncalibrated_probs.ravel(), y_val_tensor.cpu().numpy().ravel())

# Строим calibration plot
fraction_of_positives, mean_predicted_value = calibration_curve(y_val_tensor.cpu().numpy(), uncalibrated_probs,
                                                                n_bins=10)

plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Uncalibrated')

fraction_of_positives, mean_predicted_value = calibration_curve(y_val_tensor.cpu().numpy(), calibrated_probs, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, marker='s', label='Calibrated')

plt.legend()
plt.xlabel('Mean predicted value')
plt.ylabel('Fraction of positives')
plt.title('Calibration Plot')
plt.show()

import pickle

# Сохранение обученного экземпляра IsotonicRegression
with open('isotonic_regressor.pkl', 'wb') as f:
    pickle.dump(ir, f)
