"""
train_kat.py — KAT Ensemble Training (KAN + FT-Transformer)
============================================================

Полная перепись оригинального скрипта. Исправленные баги:
  1. Реальная KAN с B-сплайнами вместо фейкового MLP
  2. Сырые данные использованы последовательно (без нормализации на тензорах),
     что соответствует inference в ivf_digital_twin.py (build_nn_features → raw)
  3. Очистка inf/nan (деление на 0 в частотных признаках)
  4. BCELoss для ансамбля (не BCEWithLogitsLoss — выход уже вероятность)
  5. weight_decay вместо ручного L2 цикла
  6. predict_proba FTTransformer: [:, 1] для класса 1
  7. evaluate_model не применяет лишний sigmoid
  8. Удалён конфликт from kan import KAN + class KAN

Сохраняет файлы, совместимые с ivf_digital_twin.py:
  Prediction_KAN.pth    — веса KAN (load_state_dict)
  FTTransformer.joblib  — FTTransformer (joblib)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

# mambular FTTransformer
from mambular.models import FTTransformerClassifier

# ══════════════════════════════════════════════════════════════════════════
#  1.  НАСТОЯЩАЯ KAN (B-сплайны на рёбрах, Liu et al. 2024)
# ══════════════════════════════════════════════════════════════════════════

class KANLinear(nn.Module):
    """Один слой KAN: обучаемые B-сплайны φᵢⱼ(x) на каждом ребре i→j."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 grid_eps=0.02, grid_range=(-50.0, 50.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.scale_base   = scale_base
        self.scale_spline = scale_spline
        self.grid_eps     = grid_eps
        self.base_act     = nn.SiLU()

        # Начальная равномерная сетка; обновится через update_grid
        h    = (grid_range[1] - grid_range[0]) / grid_size
        knot = (torch.arange(-spline_order, grid_size + spline_order + 1,
                             dtype=torch.float32) * h + grid_range[0])
        self.register_buffer("grid",
                             knot.unsqueeze(0).expand(in_features, -1).contiguous())

        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self._init_weights(scale_noise)

    def _init_weights(self, scale_noise):
        nn.init.kaiming_uniform_(self.base_weight,   a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.spline_scaler, a=5 ** 0.5)
        with torch.no_grad():
            x_init = self.grid[:, self.spline_order:-self.spline_order].T
            noise = ((torch.rand(self.grid_size + 1, self.in_features,
                                 self.out_features) - 0.5)
                     * scale_noise / self.grid_size)
            self.spline_weight.data.copy_(self._curve2coeff(x_init, noise))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """B-сплайны (Кокс–де Бур); x клипируется к диапазону сетки."""
        grid = self.grid
        x_lo = grid[:, self.spline_order].unsqueeze(0)
        x_hi = grid[:, -(self.spline_order + 1)].unsqueeze(0)
        x_e  = x.clamp(x_lo, x_hi).unsqueeze(-1)
        bases = ((x_e >= grid[:, :-1]) & (x_e < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            dl = grid[:, k:-1]  - grid[:, :-(k + 1)]
            dr = grid[:, k+1:] - grid[:, 1:-k]
            bases = ((x_e - grid[:, :-(k+1)]) / (dl + 1e-8) * bases[:, :, :-1]
                     + (grid[:, k+1:] - x_e)  / (dr + 1e-8) * bases[:, :, 1:])
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Коэффициенты сплайна через псевдообратную (устойчиво к MKL SGELSY)."""
        A = torch.nan_to_num(self.b_splines(x).permute(1, 0, 2),
                             nan=0., posinf=0., neginf=0.)
        B = torch.nan_to_num(y.permute(1, 0, 2),
                             nan=0., posinf=0., neginf=0.)
        sol = torch.nan_to_num(torch.linalg.pinv(A) @ B, nan=0.)
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(self.base_act(x), self.base_weight * self.scale_base)
        spline_b = self.b_splines(x).reshape(x.size(0), -1)
        spline_w = (self.scaled_spline_weight * self.scale_spline
                    ).reshape(self.out_features, -1)
        return base_out + F.linear(spline_b, spline_w)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """Адаптирует сетку под фактическое распределение x."""
        batch = x.size(0)
        splines  = self.b_splines(x).permute(1, 0, 2)
        wt       = self.scaled_spline_weight.permute(1, 2, 0)
        y_curr   = torch.nan_to_num(
            torch.bmm(splines, wt).permute(1, 0, 2), nan=0.)

        x_sorted, _ = x.sort(dim=0)
        idx = torch.linspace(0, batch - 1, self.grid_size + 1,
                             dtype=torch.long, device=x.device)
        grid_adaptive = x_sorted[idx]
        step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform  = (torch.arange(self.grid_size + 1, dtype=x.dtype,
                                      device=x.device).unsqueeze(1) * step
                         + x_sorted[0] - margin)
        grid_new = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        prepend  = (grid_new[:1]
                    - step * torch.arange(self.spline_order, 0, -1,
                                          device=x.device).unsqueeze(1))
        append   = (grid_new[-1:]
                    + step * torch.arange(1, self.spline_order + 1,
                                          device=x.device).unsqueeze(1))
        self.grid.copy_(torch.cat([prepend, grid_new, append], dim=0).T)

        n_fit = min(batch, 512)
        self.spline_weight.data.copy_(
            self._curve2coeff(x[:n_fit], y_curr[:n_fit]))


class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network.
    KAN()                             → width=[18,10,1], grid=5, k=3
    KAN(width=[18,10,1], grid=5, k=3) → явная инициализация

    grid_range=(-50, 50): начальный широкий диапазон под сырые данные IVF;
    после вызова update_grid адаптируется под реальные признаки.
    """
    def __init__(self, width=None, grid=5, k=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 grid_eps=0.02, grid_range=(-50.0, 50.0)):
        super().__init__()
        if width is None:
            width = [18, 10, 1]
        self.width = width
        self.layers = nn.ModuleList([
            KANLinear(width[i], width[i + 1],
                      grid_size=grid, spline_order=k,
                      scale_noise=scale_noise, scale_base=scale_base,
                      scale_spline=scale_spline, grid_eps=grid_eps,
                      grid_range=grid_range)
            for i in range(len(width) - 1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(width[i + 1]) for i in range(len(width) - 2)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):
                x = self.norms[i](x)
        return x

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            layer.update_grid(x)
            x = layer(x)
            if i < len(self.norms):
                x = self.norms[i](x)


# ══════════════════════════════════════════════════════════════════════════
#  2.  АНСАМБЛЕВАЯ МОДЕЛЬ
# ══════════════════════════════════════════════════════════════════════════

class EnsembleModel(nn.Module):
    """
    KAN + FT-Transformer взвешенный ансамбль.
    Выход: вероятность (0-1), не логит.

    Обучаемые веса kan_weight / ft_weight нормализуются через softmax,
    чтобы выход оставался в [0, 1] независимо от их значений.
    """
    def __init__(self, kan_model, ft_model, feature_names):
        super().__init__()
        self.kan_model     = kan_model
        self.ft_model      = ft_model
        self.feature_names = list(feature_names)
        # Инициализируем равными весами; softmax нормализует
        self.raw_weights = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # KAN: логит → вероятность
        kan_prob = torch.sigmoid(self.kan_model(x).squeeze(-1))   # (batch,)

        # FTTransformer: предсказывает (batch, n_classes),
        # берём столбец класса 1
        x_df = pd.DataFrame(x.detach().cpu().numpy(), columns=self.feature_names)
        ft_proba = self.ft_model.predict_proba(x_df)              # (batch, 2)
        ft_prob  = torch.tensor(
            ft_proba[:, 1], dtype=torch.float32, device=x.device  # класс 1
        )

        # Нормализованные веса через softmax
        w = torch.softmax(self.raw_weights, dim=0)
        ensemble = w[0] * kan_prob + w[1] * ft_prob
        return ensemble.unsqueeze(1)                               # (batch, 1)


# ══════════════════════════════════════════════════════════════════════════
#  3.  ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════

DATA_PATH = "C:/Users/User/Desktop/IVF/AI/DNN/обучение и валидация/all_df_with_KPI.xlsx"

selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl", "Число Bl хор.кач-ва", "Частота оплодотворения",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества",
    "Частота получения ОКК", "Число эмбрионов 5 дня",
    "Заморожено эмбрионов", "Перенесено эмбрионов", "KPIScore",
]

df = pd.read_excel(DATA_PATH)

# ── Очистка ───────────────────────────────────────────────────────────────
print("=== Диагностика до очистки ===")
for col in selected_features:
    vals  = df[col].values.astype(float)
    n_inf = np.isinf(vals).sum()
    n_nan = df[col].isna().sum()
    if n_inf or n_nan:
        print(f"  {col:<52} inf={n_inf}  nan={n_nan}")

# inf / -inf → NaN
df[selected_features] = df[selected_features].replace([np.inf, -np.inf], np.nan)

# Значения вне float32 → NaN
FLOAT32_MAX = np.finfo(np.float32).max
for col in selected_features:
    mask = np.abs(df[col].values.astype(float)) > FLOAT32_MAX
    if mask.any():
        df.loc[mask, col] = np.nan

# NaN → медиана
imputer = SimpleImputer(strategy="median")
df[selected_features] = imputer.fit_transform(df[selected_features])

# Мягкое клиппирование выбросов (p1–p99)
for col in selected_features:
    lo, hi = np.percentile(df[col], 1), np.percentile(df[col], 99)
    df[col] = df[col].clip(lo, hi)

assert not np.isinf(df[selected_features].values.astype(float)).any()
assert not np.isnan(df[selected_features].values.astype(float)).any()
print(f"Данные очищены. Строк: {len(df)}\n")

# ── Разбивка ──────────────────────────────────────────────────────────────
X = df[selected_features].values.astype(np.float32)
y = df["Исход переноса"].values.astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ── Тензоры (сырые данные — согласовано с build_nn_features в app) ────────
# ВАЖНО: inference в ivf_digital_twin.py передаёт сырые данные (не нормализованные),
# поэтому обучаем на сырых данных. KAN адаптирует сетку через update_grid.
train_input = torch.FloatTensor(X_train)
val_input   = torch.FloatTensor(X_val)
test_input  = torch.FloatTensor(X_test)
train_label = torch.FloatTensor(y_train).view(-1, 1)
val_label   = torch.FloatTensor(y_val).view(-1, 1)
test_label  = torch.FloatTensor(y_test).view(-1, 1)

# DataFrame для FTTransformer
train_df = pd.DataFrame(X_train, columns=selected_features)
val_df   = pd.DataFrame(X_val,   columns=selected_features)
test_df  = pd.DataFrame(X_test,  columns=selected_features)

# ══════════════════════════════════════════════════════════════════════════
#  4.  ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════

KAN_WIDTH    = [len(selected_features), 10, 1]
GRID_SIZE    = 5
SPLINE_ORDER = 3

# KAN: grid_range=(-50, 50) — широкое начальное покрытие для сырых данных IVF.
# update_grid адаптирует сетку под реальный диапазон каждого признака.
kan_model = KAN(width=KAN_WIDTH, grid=GRID_SIZE, k=SPLINE_ORDER,
                grid_range=(-50.0, 50.0))

n_params = sum(p.numel() for p in kan_model.parameters() if p.requires_grad)
print(f"\nKAN параметры: {n_params:,}  (MLP-эквивалент: 201)")

# FTTransformer
ft_model = FTTransformerClassifier(
    d_model=64,
    n_layers=8,
    numerical_preprocessing="ple",
    n_bins=50,
)

# ══════════════════════════════════════════════════════════════════════════
#  5.  ОБУЧЕНИЕ KAN
# ══════════════════════════════════════════════════════════════════════════

KAN_EPOCHS        = 200
KAN_LR            = 1e-3
KAN_GRID_UPDATE_N = 20   # обновлять сетку каждые N эпох

# Адаптируем сетку под реальное распределение данных ДО обучения
print("\nАдаптация сетки KAN под данные...")
kan_model.update_grid(train_input)
print("Готово.\n")

kan_optimizer = torch.optim.AdamW(
    kan_model.parameters(), lr=KAN_LR, weight_decay=1e-4)
kan_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    kan_optimizer, T_max=KAN_EPOCHS)
loss_fn_bce_logits = nn.BCEWithLogitsLoss()

kan_history = {"train_loss": [], "val_loss": [], "val_auc": []}

print(f"{'Эпоха':>6}  {'TLoss':>7}  {'VLoss':>7}  {'VAUC':>6}")
print("-" * 34)

for epoch in range(1, KAN_EPOCHS + 1):
    # Обновление сетки
    if epoch % KAN_GRID_UPDATE_N == 0:
        kan_model.eval()
        with torch.no_grad():
            kan_model.update_grid(train_input)

    kan_model.train()
    kan_optimizer.zero_grad()
    logits = kan_model(train_input).squeeze()                # логиты
    loss   = loss_fn_bce_logits(logits, train_label.squeeze())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(kan_model.parameters(), 1.0)
    kan_optimizer.step()
    kan_scheduler.step()

    kan_model.eval()
    with torch.no_grad():
        vl_logits = kan_model(val_input).squeeze()
        vl_loss   = loss_fn_bce_logits(vl_logits, val_label.squeeze()).item()
        vl_probs  = torch.sigmoid(vl_logits).cpu().numpy()
        try:
            vl_auc = roc_auc_score(val_label.cpu().numpy(), vl_probs)
        except Exception:
            vl_auc = 0.5

    kan_history["train_loss"].append(loss.item())
    kan_history["val_loss"].append(vl_loss)
    kan_history["val_auc"].append(vl_auc)

    if epoch % 20 == 0 or epoch == 1:
        print(f"{epoch:6d}  {loss.item():7.4f}  {vl_loss:7.4f}  {vl_auc:6.4f}")

# ══════════════════════════════════════════════════════════════════════════
#  6.  ОБУЧЕНИЕ FTTransformer
# ══════════════════════════════════════════════════════════════════════════

print("\nОбучение FTTransformer...")
ft_model.fit(
    train_df, y_train,
    max_epochs=150,
    lr=1e-4,
    patience=10,          # early stopping
    batch_size=256,
)

# Быстрая проверка метрик FTT на валидации
ft_val_proba = ft_model.predict_proba(val_df)[:, 1]   # класс 1
ft_val_auc   = roc_auc_score(y_val, ft_val_proba)
print(f"FTTransformer Val AUC: {ft_val_auc:.4f}\n")

# ══════════════════════════════════════════════════════════════════════════
#  7.  ОБУЧЕНИЕ ВЕСОВ АНСАМБЛЯ
# ══════════════════════════════════════════════════════════════════════════
# Обучаем только raw_weights — KAN и FTT заморожены.
# BCELoss (не BCEWithLogitsLoss!): ансамбль выдаёт вероятности (0-1).

print("Обучение весов ансамбля...")

ensemble_model = EnsembleModel(kan_model, ft_model, selected_features)

# Замораживаем предобученные компоненты
for param in ensemble_model.kan_model.parameters():
    param.requires_grad_(False)

# FTTransformer — sklearn-интерфейс, параметры не в PyTorch графе

ens_optimizer = torch.optim.Adam([ensemble_model.raw_weights], lr=1e-2)
loss_fn_bce   = nn.BCELoss()   # ← вероятности, не логиты

best_val_auc  = 0.0
best_weights  = None
PATIENCE      = 10
no_improve    = 0

for epoch in range(1, 51):
    ensemble_model.train()
    ens_optimizer.zero_grad()
    out  = ensemble_model(train_input)                       # (batch, 1) вероятности
    loss = loss_fn_bce(out, train_label)
    loss.backward()
    ens_optimizer.step()

    ensemble_model.eval()
    with torch.no_grad():
        vl_out  = ensemble_model(val_input).squeeze().cpu().numpy()
        vl_auc  = roc_auc_score(y_val, vl_out)

    if vl_auc > best_val_auc:
        best_val_auc = vl_auc
        best_weights = ensemble_model.raw_weights.data.clone()
        no_improve   = 0
    else:
        no_improve += 1

    w = torch.softmax(ensemble_model.raw_weights, dim=0).detach()
    if epoch % 5 == 0:
        print(f"  Эпоха {epoch:3d}: Val AUC={vl_auc:.4f} "
              f"[w_KAN={w[0]:.3f}  w_FT={w[1]:.3f}]")

    if no_improve >= PATIENCE:
        print(f"  Early stopping на эпохе {epoch}")
        break

# Восстанавливаем лучшие веса
if best_weights is not None:
    ensemble_model.raw_weights.data.copy_(best_weights)

w_final = torch.softmax(ensemble_model.raw_weights, dim=0).detach()
print(f"\nФинальные веса: KAN={w_final[0]:.3f}  FT={w_final[1]:.3f}")
print(f"Лучший Val AUC ансамбля: {best_val_auc:.4f}\n")

# ══════════════════════════════════════════════════════════════════════════
#  8.  ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ
# ══════════════════════════════════════════════════════════════════════════

def evaluate(model_fn, X_tensor, y_true_np, label):
    """model_fn(X_tensor) → вероятности (0-1) в numpy."""
    probs = model_fn(X_tensor)
    preds = (probs > 0.5).astype(int)
    y_int = y_true_np.astype(int)
    print(f"\n── {label} ─────────────────────────────────")
    print(f"  Accuracy:   {accuracy_score(y_int, preds):.4f}")
    print(f"  Precision:  {precision_score(y_int, preds, zero_division=0):.4f}")
    print(f"  Recall:     {recall_score(y_int, preds, zero_division=0):.4f}")
    print(f"  F1-score:   {f1_score(y_int, preds, zero_division=0):.4f}")
    print(f"  ROC-AUC:    {roc_auc_score(y_int, probs):.4f}")
    return probs

# KAN
kan_model.eval()
with torch.no_grad():
    kan_test_fn = lambda x: torch.sigmoid(
        kan_model(x).squeeze()).cpu().numpy()
kan_probs = evaluate(kan_test_fn, test_input, y_test, "KAN")

# FTTransformer
ft_probs = ft_model.predict_proba(test_df)[:, 1]
evaluate(lambda _: ft_probs, test_input, y_test, "FTTransformer")

# Ensemble
ensemble_model.eval()
with torch.no_grad():
    ens_fn = lambda x: ensemble_model(x).squeeze().cpu().numpy()
ens_probs = evaluate(ens_fn, test_input, y_test, "Ensemble")

# ══════════════════════════════════════════════════════════════════════════
#  9.  ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════

# KAN обучение
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(kan_history["train_loss"], label="Train"); axes[0].plot(kan_history["val_loss"], label="Val")
axes[0].set_title("KAN Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(kan_history["val_auc"])
axes[1].set_title("KAN Val AUC"); axes[1].grid(True, alpha=0.3)

# ROC кривые
for name, probs in [("KAN", kan_probs), ("FTT", ft_probs), ("Ensemble", ens_probs)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[2].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
axes[2].plot([0,1],[0,1],"k--",lw=1)
axes[2].set_title("ROC — Test"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kat_training.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
#  10. КАЛИБРОВКА АНСАМБЛЯ (изотоническая регрессия)
# ══════════════════════════════════════════════════════════════════════════

ensemble_model.eval()
with torch.no_grad():
    val_probs_ens = ensemble_model(val_input).squeeze().cpu().numpy()

ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(val_probs_ens, y_val)
calib_test = ir.predict(ens_probs)

# Calibration plot
fig, ax = plt.subplots(figsize=(6, 5))
fop_u, mpv_u = calibration_curve(y_test, ens_probs,   n_bins=10)
fop_c, mpv_c = calibration_curve(y_test, calib_test, n_bins=10)
ax.plot([0,1],[0,1],"k--",label="Perfect")
ax.plot(mpv_u, fop_u, "o-", label="Uncalibrated")
ax.plot(mpv_c, fop_c, "s-", label="Isotonic calibrated")
ax.set_title("Calibration — Ensemble"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kat_calibration.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
#  11. СОХРАНЕНИЕ ФАЙЛОВ
#      Имена совместимы с NN_MODEL_PATHS в ivf_digital_twin.py
# ══════════════════════════════════════════════════════════════════════════

# 1. Веса KAN → Prediction_KAN.pth
torch.save(kan_model.state_dict(), "Prediction_KAN.pth")
print("[OK] Prediction_KAN.pth")

# 2. FTTransformer → FTTransformer.joblib
joblib.dump(ft_model, "FTTransformer.joblib")
print("[OK] FTTransformer.joblib")

# 3. Калибровщик (отдельно, опционально)
with open("isotonic_ensemble.pkl", "wb") as f:
    pickle.dump(ir, f)
print("[OK] isotonic_ensemble.pkl")

print(f"\nГотово. Test Ensemble AUC: {roc_auc_score(y_test, ens_probs):.4f}")

# ══════════════════════════════════════════════════════════════════════════
#  ПРИМЕЧАНИЕ: KAT_calibrated_model.pkl
# ══════════════════════════════════════════════════════════════════════════
# ivf_digital_twin.py при загрузке пробует:
#   wrapped = joblib.load('KAT_calibrated_model.pkl')
# и при неудаче автоматически падает обратно на EnsembleWrapper(ensemble).
# Поэтому KAT_calibrated_model.pkl не создаётся здесь намеренно:
# класс EnsembleWrapper определён в ivf_digital_twin.py и не импортируется
# в этот скрипт — pickle не смог бы найти класс при десериализации.
# Для получения KAT_calibrated_model.pkl запустите patch_ivf_digital_twin.py
# (который обновляет класс KAN), после чего система использует
# EnsembleWrapper(ensemble) автоматически при загрузке.
