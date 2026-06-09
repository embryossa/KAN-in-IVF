"""
Реализует KAN согласно Liu et al. 2024 («KAN: Kolmogorov-Arnold Networks»).

Ключевое отличие от MLP:
  MLP:  активации на УЗЛАХ (фиксированный ReLU/tanh)
  KAN:  активации на РЁБРАХ (обучаемые B-сплайны φᵢⱼ(x))

Формула слоя KAN:
  yⱼ = Σᵢ φᵢⱼ(xᵢ)
  φᵢⱼ(x) = scale_base · silu(x) · w_base
           + scale_spline · Σₛ cᵢⱼₛ · Bₛ(x)

  где Bₛ — B-сплайн базисные функции (порядок k=3, Кокс–де Бур),
  cᵢⱼₛ — обучаемые коэффициенты сплайна.

После обучения:
  - Сохраняет Prediction_KAN.pth (совместим с ivf_digital_twin.py)
  - Сохраняет isotonic_regressor_new.pkl (калибровка)

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import pickle

# ══════════════════════════════════════════════════════════════════════════
#  1.  НАСТОЯЩАЯ KAN: B-СПЛАЙНОВЫЕ ФУНКЦИИ НА РЁБРАХ
# ══════════════════════════════════════════════════════════════════════════

class KANLinear(nn.Module):
    """
    Один слой KAN: in_features → out_features.

    Каждое ребро i→j содержит обучаемую функцию:
        φᵢⱼ(x) = w_base · silu(x)  +  Σₛ cᵢⱼₛ · Bₛ(x)

    Выход слоя:
        yⱼ = Σᵢ φᵢⱼ(xᵢ)

    Параметры:
        grid_size    — число интервалов B-сплайна (G)
        spline_order — порядок B-сплайна (k); k=3 → кубические сплайны
        Каждый сплайн имеет G+k базисных функций.

    Сетка (grid):
        Хранится как буфер (in_features, G + 2k + 1).
        Может обновляться через update_grid() под распределение данных.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.scale_base   = scale_base
        self.scale_spline = scale_spline
        self.grid_eps     = grid_eps
        self.base_act     = nn.SiLU()

        # ── Инициализация сетки ──────────────────────────────────────────
        # Равномерная сетка с расширенными граничными узлами:
        # shape (in_features, grid_size + 2*spline_order + 1)
        h    = (grid_range[1] - grid_range[0]) / grid_size
        knot = torch.arange(
            -spline_order, grid_size + spline_order + 1, dtype=torch.float32
        ) * h + grid_range[0]                      # (G + 2k + 1,)
        grid = knot.unsqueeze(0).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)          # не обучается, но сохраняется

        # ── Обучаемые параметры ──────────────────────────────────────────
        # Базовые (линейные) веса: SiLU-путь
        self.base_weight  = nn.Parameter(torch.empty(out_features, in_features))
        # Коэффициенты сплайнов: (out, in, G+k)
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        # Скалярный масштаб каждого ребра (улучшает сходимость)
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self._init_weights(scale_noise)

    # ── Инициализация весов ──────────────────────────────────────────────

    def _init_weights(self, scale_noise: float):
        nn.init.kaiming_uniform_(self.base_weight,   a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.spline_scaler, a=5 ** 0.5)
        with torch.no_grad():
            # Инициализируем сплайн-веса через небольшой шум на узлах сетки
            x_init = self.grid[:, self.spline_order:-self.spline_order].T
            # x_init: (grid_size+1, in_features) — внутренние узлы
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * scale_noise / self.grid_size
            self.spline_weight.data.copy_(self._curve2coeff(x_init, noise))

    # ── B-сплайновые базисные функции (Кокс–де Бур) ─────────────────────

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет B-сплайновые базисные значения для входа x.

        Рекуррентная формула Кокса–де Бура:
            B[i,0](x) = 1 если t[i] ≤ x < t[i+1], иначе 0
            B[i,k](x) = (x - t[i]) / (t[i+k] - t[i]) * B[i,k-1](x)
                       + (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * B[i+1,k-1](x)

        Args:
            x: (batch, in_features)
        Returns:
            bases: (batch, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features, \
            f"Ожидается (batch, {self.in_features}), получено {x.shape}"

        grid = self.grid        # (in_features, G + 2k + 1)
        x_e  = x.unsqueeze(-1) # (batch, in_features, 1)  для broadcast

        # Порядок 0: индикаторные функции на интервалах сетки
        bases = ((x_e >= grid[:, :-1]) & (x_e < grid[:, 1:])).to(x.dtype)
        # bases: (batch, in_features, G + 2k)

        # Рекурсия по порядку k = 1, 2, ..., spline_order
        for k in range(1, self.spline_order + 1):
            denom_l = grid[:, k:-1]   - grid[:, :-(k+1)]   # (in, n_bases+1)
            denom_r = grid[:, k+1:]   - grid[:, 1:-k]       # (in, n_bases+1)
            left  = (x_e - grid[:, :-(k+1)]) / (denom_l + 1e-8) * bases[:, :, :-1]
            right = (grid[:, k+1:] - x_e)    / (denom_r + 1e-8) * bases[:, :, 1:]
            bases = left + right
            # После шага k: bases имеет (G + 2k - k) = (G + k) базисных функций

        assert bases.shape == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    # ── МНК-подгонка коэффициентов под контрольные точки ────────────────

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет коэффициенты сплайна по парам (x, y) через МНК.

        Args:
            x: (n_samples, in_features)  — точки оценки
            y: (n_samples, in_features, out_features) — целевые значения
        Returns:
            coeff: (out_features, in_features, grid_size + spline_order)
        """
        A = self.b_splines(x)           # (n, in, G+k)
        A = A.permute(1, 0, 2)          # (in, n, G+k)
        B = y.permute(1, 0, 2)          # (in, n, out)

        # Решение переопределённой/недоопределённой системы A @ coeff = B
        solution = torch.linalg.lstsq(A, B).solution  # (in, G+k, out)
        return solution.permute(2, 0, 1).contiguous()  # (out, in, G+k)

    # ── Масштабированные веса сплайна ────────────────────────────────────

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """Веса сплайна × индивидуальный скалярный масштаб ребра."""
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    # ── Прямой проход ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_features)
        Returns: (batch, out_features)

        Два пути суммируются:
          1. Базовый (SiLU + линейный) — быстрый глобальный отклик
          2. Сплайновый — локальный точный отклик через B-сплайны
        """
        # Путь 1: SiLU(x) @ base_weight.T
        base_out = F.linear(self.base_act(x), self.base_weight * self.scale_base)

        # Путь 2: [basis_1, ..., basis_{in·(G+k)}] @ spline_weight_flat.T
        spline_basis = self.b_splines(x).reshape(x.size(0), -1)   # (batch, in*(G+k))
        spline_wt    = (self.scaled_spline_weight * self.scale_spline)
        spline_out   = F.linear(spline_basis, spline_wt.reshape(self.out_features, -1))

        return base_out + spline_out

    # ── Адаптивное обновление сетки ──────────────────────────────────────

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Обновляет B-сплайновую сетку под фактическое распределение входных данных.
        После обновления перефитирует коэффициенты, сохраняя функцию.

        Вызывать: перед обучением и каждые N эпох.
        Args:
            x:      (batch, in_features) — репрезентативная выборка
            margin: отступ за границы min/max
        """
        batch = x.size(0)

        # ── Вычислить текущие выходы сплайна (до обновления сетки) ───────
        splines = self.b_splines(x).permute(1, 0, 2)    # (in, batch, G+k)
        wt      = self.scaled_spline_weight.permute(1, 2, 0)  # (in, G+k, out)
        y_curr  = torch.bmm(splines, wt).permute(1, 0, 2)    # (batch, in, out)

        # ── Построить новую сетку из распределения данных ─────────────────
        x_sorted, _ = x.sort(dim=0)   # (batch, in_features)

        # Квантильная (адаптивная) сетка
        idx = torch.linspace(0, batch - 1, self.grid_size + 1,
                             dtype=torch.long, device=x.device)
        grid_adaptive = x_sorted[idx]          # (G+1, in)

        # Равномерная сетка по диапазону данных
        step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
            * step + x_sorted[0] - margin
        )                                       # (G+1, in)

        # Смешиваем: grid_eps — доля равномерной сетки
        grid_new = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # Добавляем граничные узлы с обеих сторон
        prepend = (grid_new[:1]
                   - step * torch.arange(self.spline_order, 0, -1,
                                         device=x.device).unsqueeze(1))
        append  = (grid_new[-1:]
                   + step * torch.arange(1, self.spline_order + 1,
                                         device=x.device).unsqueeze(1))
        grid_new = torch.cat([prepend, grid_new, append], dim=0)  # (G+2k+1, in)

        self.grid.copy_(grid_new.T)   # (in, G+2k+1)

        # ── Перефитировать коэффициенты, сохраняя выученную функцию ──────
        # Используем x (обучающий батч) как контрольные точки
        n_fit = min(batch, 512)
        new_coeff = self._curve2coeff(x[:n_fit], y_curr[:n_fit])
        self.spline_weight.data.copy_(new_coeff)


# ══════════════════════════════════════════════════════════════════════════
#  2.  ПОЛНАЯ KAN: СТЕК СЛОЁВ
# ══════════════════════════════════════════════════════════════════════════

class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network.

    Интерфейс совместим с оригинальным кодом обучения:
        model = KAN(width=[18, 10, 1], grid=5, k=3)
        model = KAN()   ← использует дефолтные параметры для IVF DT

    Параметры:
        width  — список размерностей [in, hidden, ..., out]
        grid   — число интервалов B-сплайна (grid_size)
        k      — порядок B-сплайна (spline_order); k=3 → кубические

    Между слоями применяется LayerNorm — стабилизирует обучение
    и поддерживает входы близко к области сетки [-1, 1].
    """

    def __init__(
        self,
        width: list  = None,
        grid:  int   = 5,
        k:     int   = 3,
        scale_noise:  float = 0.1,
        scale_base:   float = 1.0,
        scale_spline: float = 1.0,
        grid_eps:     float = 0.02,
        grid_range:   tuple = (-1.0, 1.0),
    ):
        super().__init__()

        # Дефолтные параметры для IVF Digital Twin (18 признаков)
        if width is None:
            width = [18, 10, 1]

        assert len(width) >= 2, "width должен содержать минимум [in, out]"

        self.width        = width
        self.grid_size    = grid
        self.spline_order = k

        # Слои KAN
        self.layers = nn.ModuleList([
            KANLinear(
                in_features=width[i],
                out_features=width[i + 1],
                grid_size=grid,
                spline_order=k,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            for i in range(len(width) - 1)
        ])

        # LayerNorm после каждого скрытого слоя (кроме последнего)
        self.norms = nn.ModuleList([
            nn.LayerNorm(width[i + 1])
            for i in range(len(width) - 2)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):        # нормализация после скрытых слоёв
                x = self.norms[i](x)
        return x

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor):
        """
        Обновить сетки всех слоёв последовательно.
        x — входной батч (обучающие данные).
        """
        for i, layer in enumerate(self.layers):
            layer.update_grid(x)
            x = layer(x)                   # пропускаем через обновлённый слой
            if i < len(self.norms):
                x = self.norms[i](x)


# ══════════════════════════════════════════════════════════════════════════
#  3.  ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════

df = pd.read_excel("C:/Users/User/Desktop/IVF/AI/DNN/обучение и валидация/all_df_with_KPI.xlsx")

selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl хор.кач-ва", "Частота оплодотворения", "Число Bl",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества",
    "Частота получения ОКК", "Число эмбрионов 5 дня",
    "Заморожено эмбрионов", "Перенесено эмбрионов", "KPIScore",
]

X = df[selected_features].values.astype(np.float32)
y = df["Исход переноса"].values.astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s   = scaler.transform(X_val).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

dataset = {
    "train_input": torch.from_numpy(X_train_s),
    "val_input":   torch.from_numpy(X_val_s),
    "test_input":  torch.from_numpy(X_test_s),
    "train_label": torch.from_numpy(y_train),
    "val_label":   torch.from_numpy(y_val),
    "test_label":  torch.from_numpy(y_test),
}

print(f"Обучающая выборка:   {X_train_s.shape[0]} примеров")
print(f"Валидационная:       {X_val_s.shape[0]} примеров")
print(f"Тестовая:            {X_test_s.shape[0]} примеров")
print(f"Признаков:           {X_train_s.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════
#  4.  ИНИЦИАЛИЗАЦИЯ МОДЕЛИ И ОПТИМИЗАТОРА
# ══════════════════════════════════════════════════════════════════════════

GRID_SIZE    = 5   # число интервалов B-сплайна
SPLINE_ORDER = 3   # порядок сплайна (кубические)
HIDDEN_DIM   = 10  # размер скрытого слоя
N_EPOCHS     = 200 # общее число эпох
LR           = 1e-3
GRID_UPDATE_EVERY = 20  # обновлять сетку каждые N эпох

model     = KAN(width=[18, HIDDEN_DIM, 1], grid=GRID_SIZE, k=SPLINE_ORDER)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
loss_fn   = nn.BCEWithLogitsLoss()

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nПараметры KAN:  {n_params:,}")
print(f"  (MLP 18→10→1 имел бы: {18*10 + 10 + 10*1 + 1:,} параметров)\n")

# ── Первоначальная инициализация сетки под обучающие данные ──────────────
# Это ключевая особенность KAN: сетка адаптируется к данным, а не к [-1,1]
print("Инициализация адаптивной сетки на обучающих данных...")
model.update_grid(dataset["train_input"])
print("Сетка инициализирована.\n")

# ══════════════════════════════════════════════════════════════════════════
#  5.  ЦИКЛ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════════

history = {k: [] for k in [
    "train_loss", "val_loss",
    "train_acc",  "val_acc",
    "train_f1",   "val_f1",
]}

print(f"{'Эпоха':>6}  {'TLoss':>7}  {'VLoss':>7}  "
      f"{'TAcc':>6}  {'VAcc':>6}  {'TF1':>6}  {'VF1':>6}")
print("-" * 58)

for epoch in range(1, N_EPOCHS + 1):

    # ── Обновление сетки каждые GRID_UPDATE_EVERY эпох ───────────────────
    # Сетка подстраивается под текущее распределение активаций,
    # что позволяет сплайнам сосредоточиться на плотных областях данных
    if epoch % GRID_UPDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            model.update_grid(dataset["train_input"])

    # ── Шаг обучения ─────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    logits = model(dataset["train_input"]).squeeze()
    loss   = loss_fn(logits, dataset["train_label"])
    loss.backward()

    # Gradient clipping — KAN с большим числом параметров может иметь
    # нестабильные градиенты в начале обучения
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # ── Метрики ───────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        # Обучающий набор
        tr_preds = (torch.sigmoid(logits) > 0.5).long()
        tr_acc   = (tr_preds == dataset["train_label"].long()).float().mean().item()
        tr_f1    = f1_score(dataset["train_label"].cpu(), tr_preds.cpu(), zero_division=0)

        # Валидационный набор
        vl_logits = model(dataset["val_input"]).squeeze()
        vl_loss   = loss_fn(vl_logits, dataset["val_label"]).item()
        vl_preds  = (torch.sigmoid(vl_logits) > 0.5).long()
        vl_acc    = (vl_preds == dataset["val_label"].long()).float().mean().item()
        vl_f1     = f1_score(dataset["val_label"].cpu(), vl_preds.cpu(), zero_division=0)

    history["train_loss"].append(loss.item())
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["train_f1"].append(tr_f1)
    history["val_f1"].append(vl_f1)

    if epoch % 10 == 0 or epoch == 1:
        grid_marker = " ← grid updated" if epoch % GRID_UPDATE_EVERY == 0 else ""
        print(f"{epoch:6d}  {loss.item():7.4f}  {vl_loss:7.4f}  "
              f"{tr_acc:6.4f}  {vl_acc:6.4f}  {tr_f1:6.4f}  {vl_f1:6.4f}"
              f"{grid_marker}")

# ══════════════════════════════════════════════════════════════════════════
#  6.  ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ
# ══════════════════════════════════════════════════════════════════════════

model.eval()
with torch.no_grad():
    test_logits = model(dataset["test_input"]).squeeze()
    test_loss   = loss_fn(test_logits, dataset["test_label"]).item()
    test_scores = torch.sigmoid(test_logits).cpu().numpy()
    test_preds  = (torch.tensor(test_scores) > 0.5).long()

test_acc  = (test_preds == dataset["test_label"].long()).float().mean().item()
test_prec = precision_score(dataset["test_label"].cpu(), test_preds, zero_division=0)
test_rec  = recall_score(dataset["test_label"].cpu(), test_preds, zero_division=0)
test_f1   = f1_score(dataset["test_label"].cpu(), test_preds, zero_division=0)

from sklearn.metrics import roc_auc_score
test_auc  = roc_auc_score(dataset["test_label"].cpu().numpy(), test_scores)

print(f"\n{'='*50}")
print(f"  ТЕСТОВЫЕ РЕЗУЛЬТАТЫ")
print(f"{'='*50}")
print(f"  Loss:       {test_loss:.4f}")
print(f"  Accuracy:   {test_acc:.4f}")
print(f"  Precision:  {test_prec:.4f}")
print(f"  Recall:     {test_rec:.4f}")
print(f"  F1-score:   {test_f1:.4f}")
print(f"  ROC-AUC:    {test_auc:.4f}")
print(f"{'='*50}\n")

# ══════════════════════════════════════════════════════════════════════════
#  7.  ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Loss
ax = axes[0]
ax.plot(history["train_loss"], label="Train Loss")
ax.plot(history["val_loss"],   label="Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Training and Validation Loss")
ax.legend(); ax.grid(True, alpha=0.3)

# Accuracy
ax = axes[1]
ax.plot(history["train_acc"], label="Train Acc")
ax.plot(history["val_acc"],   label="Val Acc")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Training and Validation Accuracy")
ax.legend(); ax.grid(True, alpha=0.3)

# F1
ax = axes[2]
ax.plot(history["train_f1"], label="Train F1")
ax.plot(history["val_f1"],   label="Val F1")
ax.set_xlabel("Epoch"); ax.set_ylabel("F1-score")
ax.set_title("F1-score over Training")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kan_training_curves.png", dpi=150)
plt.show()


def plot_roc_prc(y_true, y_scores, title_suffix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC — {title_suffix}")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    prc_auc = auc(rec, prec)
    axes[1].plot(rec, prec, color="steelblue", lw=2,
                 label=f"PRC AUC = {prc_auc:.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall — {title_suffix}")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"kan_roc_prc_{title_suffix.lower()}.png", dpi=150)
    plt.show()


model.eval()
with torch.no_grad():
    val_scores = torch.sigmoid(model(dataset["val_input"]).squeeze()).cpu().numpy()
plot_roc_prc(dataset["val_label"].cpu().numpy(), val_scores, "Validation")
plot_roc_prc(dataset["test_label"].cpu().numpy(), test_scores, "Test")

# ══════════════════════════════════════════════════════════════════════════
#  8.  КАЛИБРОВКА (изотоническая регрессия, как в оригинале)
# ══════════════════════════════════════════════════════════════════════════

model.eval()
with torch.no_grad():
    val_logits_cal = model(dataset["val_input"]).squeeze()
    uncalib_probs  = torch.sigmoid(val_logits_cal).cpu().numpy()

ir = IsotonicRegression(out_of_bounds="clip")
calib_probs = ir.fit_transform(
    uncalib_probs.ravel(),
    dataset["val_label"].cpu().numpy().ravel()
)

# Calibration plot
fig, ax = plt.subplots(figsize=(7, 5))
fop_uc, mpv_uc = calibration_curve(
    dataset["val_label"].cpu().numpy(), uncalib_probs, n_bins=10
)
fop_c, mpv_c = calibration_curve(
    dataset["val_label"].cpu().numpy(), calib_probs, n_bins=10
)
ax.plot([0, 1], [0, 1], "k--", label="Perfect")
ax.plot(mpv_uc, fop_uc, "o-", label="Uncalibrated KAN")
ax.plot(mpv_c,  fop_c,  "s-", label="Isotonic calibrated")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Plot — KAN")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kan_calibration.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
#  9.  СОХРАНЕНИЕ ФАЙЛОВ
# ══════════════════════════════════════════════════════════════════════════

# Сохраняем веса модели (совместимо с load_state_dict в ivf_digital_twin.py
# ПОСЛЕ того как там обновлён класс KAN — см. PATCH ниже)
torch.save(model.state_dict(), "Prediction_KAN.pth")
print("[OK] Prediction_KAN.pth сохранён")

# Калибровщик (isotonic regression — формат как в оригинале)
with open("isotonic_regressor_new.pkl", "wb") as f:
    pickle.dump(ir, f)
print("[OK] isotonic_regressor_new.pkl сохранён")

print("\nОбучение завершено.")
print(f"Параметров в реальной KAN: {n_params:,}")
print(f"ROC-AUC на тесте: {test_auc:.4f}")
