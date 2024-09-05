import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import pickle

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

kan_model = KAN(width=[X_train_normalized.shape[1], 10, 1], grid=5, k=3)  # Для бинарной классификации выходной слой имеет размер 1

# Определение оптимизатора и функции потерь
optimizer = torch.optim.Adam(kan_model.parameters())
loss_fn = torch.nn.BCEWithLogitsLoss()

# Обучение модели KAN
for step in range(100):
    kan_model.train()
    optimizer.zero_grad()

    outputs = kan_model(dataset['train_input']).squeeze()
    logits = outputs
    loss = loss_fn(logits, dataset['train_label'])
    loss.backward()
    optimizer.step()

# Получение предсказаний от модели KAN
kan_model.eval()
with torch.no_grad():
    val_outputs_kan = torch.sigmoid(kan_model(dataset['val_input'])).cpu().numpy()
    test_outputs_kan = torch.sigmoid(kan_model(dataset['test_input'])).cpu().numpy()

# Изменение формата данных для обучения модели DNN
X_train_reshaped = X_train_normalized.reshape((X_train_normalized.shape[0], X_train_normalized.shape[1], 1))
X_val_reshaped = X_val_normalized.reshape((X_val_normalized.shape[0], X_val_normalized.shape[1], 1))
X_test_reshaped = X_test_normalized.reshape((X_test_normalized.shape[0], X_test_normalized.shape[1], 1))

# Создание модели DNN
def create_regularized_model():
    model = Sequential()
    model.add(SimpleRNN(32, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]),
                        return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(SimpleRNN(16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))  # Применение Dropout для уменьшения переобучения
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(learning_rate=0.00001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

dnn_model = create_regularized_model()
epochs = 12
batch_size = 8

# Обучение модели DNN
dnn_model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_reshaped, y_val))

# Получение предсказаний от модели DNN
val_outputs_dnn = dnn_model.predict(X_val_reshaped).ravel()
test_outputs_dnn = dnn_model.predict(X_test_reshaped).ravel()

# Создание метамодели (логистическая регрессия) для стэкинга
stacked_val_features = np.column_stack((val_outputs_kan, val_outputs_dnn))
stacked_test_features = np.column_stack((test_outputs_kan, test_outputs_dnn))

meta_model = LogisticRegression()
meta_model.fit(stacked_val_features, y_val)

# Оценка метамодели на тестовых данных
meta_preds = meta_model.predict_proba(stacked_test_features)[:, 1]
meta_preds_labels = meta_model.predict(stacked_test_features)

test_accuracy = meta_model.score(stacked_test_features, y_test)

# Построение графиков ROC-AUC и PRC
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

plot_roc_prc(y_test, meta_preds, 'Test')

# Построение графика калибровки
def plot_calibration_curve(y_true, y_probs, dataset_type):
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Meta model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration curve - {dataset_type}')
    plt.legend(loc="best")
    plt.show()

plot_calibration_curve(y_test, meta_preds, 'Test')

# Вывод метрик качества
precision = precision_score(y_test, meta_preds_labels)
recall = recall_score(y_test, meta_preds_labels)
f1 = f1_score(y_test, meta_preds_labels)
roc_auc = roc_auc_score(y_test, meta_preds)
mcc = matthews_corrcoef(y_test, meta_preds_labels)

print(f'Accuracy: {test_accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC-AUC: {roc_auc:.2f}')
print(f'MCC: {mcc:.2f}')

# Сохранение метамодели в файл
#with open('meta_model.pkl', 'wb') as f:
    #pickle.dump(meta_model, f)

print(f'Точность метамодели на тестовых данных: {test_accuracy}')

from sklearn.metrics import brier_score_loss
from scipy.stats import chi2

# Расчет Brier Score
brier = brier_score_loss(y_test, meta_preds)
print(f'Brier Score: {brier:.4f}')

# Расчет Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_probs, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

ece = expected_calibration_error(y_test, meta_preds, n_bins=10)
print(f'Expected Calibration Error (ECE): {ece:.4f}')

# Расчет Maximum Calibration Error (MCE)
def maximum_calibration_error(y_true, y_probs, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

    return mce

mce = maximum_calibration_error(y_test, meta_preds, n_bins=10)
print(f'Maximum Calibration Error (MCE): {mce:.4f}')

# Тест Hosmer-Lemeshow
def hosmer_lemeshow_test(y_true, y_probs, n_groups=10):
    data = pd.DataFrame({'true': y_true, 'probs': y_probs})
    data['group'] = pd.qcut(data['probs'], n_groups, duplicates='drop')

    observed = data.groupby('group')['true'].sum()
    expected = data.groupby('group')['probs'].sum()
    total = data.groupby('group').size()

    hl_stat = ((observed - expected) ** 2 / (expected * (1 - expected / total))).sum()
    p_value = 1 - chi2.cdf(hl_stat, df=n_groups - 2)

    return hl_stat, p_value

hl_stat, hl_p_value = hosmer_lemeshow_test(y_test, meta_preds, n_groups=10)
print(f'Hosmer-Lemeshow Test Statistic: {hl_stat:.4f}')
print(f'Hosmer-Lemeshow p-value: {hl_p_value:.4f}')

# Вывод всех метрик
print("\nCalibration Metrics:")
print(f'Brier Score: {brier:.4f}')
print(f'Expected Calibration Error (ECE): {ece:.4f}')
print(f'Maximum Calibration Error (MCE): {mce:.4f}')
print(f'Hosmer-Lemeshow Test Statistic: {hl_stat:.4f}, p-value: {hl_p_value:.4f}')
