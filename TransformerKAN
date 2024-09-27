import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
from kan import KAN
from mambular.models import FTTransformerClassifier

# Загрузка данных
df = pd.read_excel("C:/Users/User/Desktop/IVF/AI/DNN/обучение и валидация/all_df_with_KPI.xlsx")

# Выбор признаков и целевой переменной
selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl", "Число Bl хор.кач-ва", "Частота оплодотворения",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов",
    "KPIScore"
]

X = df[selected_features].values
y = df['Исход переноса'].values

# Разделение на обучающий, валидационный и тестовый наборы
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

# Convert to tensors
train_input = torch.FloatTensor(X_train)
val_input = torch.FloatTensor(X_val)
test_input = torch.FloatTensor(X_test)
train_label = torch.FloatTensor(y_train).view(-1, 1)
val_label = torch.FloatTensor(y_val).view(-1, 1)
test_label = torch.FloatTensor(y_test).view(-1, 1)


class KAN(nn.Module):
    def __init__(self, width, grid, k):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(width[0], width[1])
        self.fc2 = nn.Linear(width[1], width[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the EnsembleModel
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


# Инициализация моделей
kan_model = KAN(width=[X_train.shape[1], 10, 1], grid=5, k=3)
lamb = 0.001 # Коэффициент L2 регуляризации
lamb_entropy = 0.01  # Коэффициент энтропийной регуляризации

ft_model = FTTransformerClassifier(
    d_model=64,
    n_layers=8,
    numerical_preprocessing="ple",
    n_bins=50
)

ensemble_model = EnsembleModel(kan_model, ft_model, feature_names=df[selected_features].columns)

# Функция обучения ансамблевой модели
def train_ensemble(model, train_input, train_label, val_input, val_label, epochs=30, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_acc = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_input)
        loss = loss_fn(outputs, train_label)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_input)
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            val_accuracy = (val_preds == val_label).float().mean().item()

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            no_improve = 0
            torch.save(model.state_dict(), 'best_ensemble_model.pth')
        else:
            no_improve += 1

        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Ensemble Val Acc={val_accuracy:.4f}, Loss={loss.item():.4f}')

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load('best_ensemble_model.pth'))


# Define the loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=1e-5, weight_decay=1e-3)

# Train the KAN model
for step in range(100):
    kan_model.train()
    optimizer.zero_grad()
    outputs = kan_model(train_input).squeeze()
    logits = outputs

    loss = loss_fn(logits, train_label.squeeze())
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in kan_model.parameters():
        l2_reg = l2_reg + torch.norm(param, 2)
    loss += lamb * l2_reg

    entropy = -F.log_softmax(logits, dim=0).mean()
    loss -= lamb_entropy * entropy
    loss.backward()
    optimizer.step()

# Train the FTTransformer model
ft_model.fit(pd.DataFrame(X_train, columns=selected_features), y_train, max_epochs=150, lr=1e-4)

# Train the EnsembleModel with early stopping
train_ensemble(ensemble_model, train_input, train_label, val_input, val_label)


# Функция оценки модели
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        # Убедимся, что выходные данные имеют правильную форму
        outputs = outputs.squeeze()
        preds = (outputs > 0.5).float()
        acc = accuracy_score(y.cpu(), preds.cpu())
        prec = precision_score(y.cpu(), preds.cpu())
        recall = recall_score(y.cpu(), preds.cpu())
        f1 = f1_score(y.cpu(), preds.cpu())
        roc_auc = roc_auc_score(y.cpu(), outputs.cpu())
    return acc, prec, recall, f1, roc_auc

# Оценка моделей
kan_results = evaluate_model(kan_model, test_input, test_label)

# Для FTTransformer
ft_df = pd.DataFrame(X_test, columns=selected_features)
ft_preds = ft_model.predict(ft_df)
ft_proba = ft_model.predict_proba(ft_df)
ft_results = (
    accuracy_score(y_test, ft_preds),
    precision_score(y_test, ft_preds),
    recall_score(y_test, ft_preds),
    f1_score(y_test, ft_preds),
    roc_auc_score(y_test, ft_proba)
)

ensemble_results = evaluate_model(ensemble_model, test_input, test_label)

# Вывод результатов
models = ['KAN', 'FTTransformer', 'Ensemble']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
results = pd.DataFrame([kan_results, ft_results, ensemble_results],
                       columns=metrics, index=models)
print(results)
results.to_csv("model_results.csv")

# Visualize the ROC and PRC for the test dataset
def plot_roc_prc(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prc_auc = auc(recall, precision)

    plt.figure(figsize=(12, 5))

    # ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc="lower right")

    # Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PRC - {model_name}')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

# Visualize the ROC and PRC for the ensemble model on the test dataset
ensemble_model.eval()
with torch.no_grad():
    ensemble_probs = ensemble_model(test_input).detach().cpu().numpy()

ensemble_probs = ensemble_probs.squeeze()
y_true = test_label.cpu().numpy()

plot_roc_prc(y_true, ensemble_probs, "Ensemble Model")


# Сохранение ансамблевой модели
torch.save(ensemble_model.state_dict(), 'KAT.pth')
