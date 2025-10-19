import numpy as np
import itertools
import matplotlib.pyplot as plt

# --------------------- Вхідні дані ---------------------
W = np.array([
    [0.2, 0.3, 0.3],
    [0.4, 0.3, 0.3],
    [0.4, 0.2, 0.4]
], dtype=float)

w_q = np.array([0.5, 0.35, 0.25], dtype=float)

# --------------------- Перевірки ---------------------
n, m = W.shape
if n < 2:
    raise ValueError("Матриця W повинна містити щонайменше 2 моделі (рядки).")

if np.any(W < 0):
    raise ValueError("Матриця W не може містити від’ємних значень.")

col_sum = W.sum(axis=0, keepdims=True)  # (1, m)
if np.any(col_sum == 0):
    raise ValueError("У матриці W є стовпці з сумою 0. Неможливо нормувати.")

if np.any(w_q < 0):
    raise ValueError("Вектор ваг w_q не може містити від’ємних значень.")

s = w_q.sum()
if np.isclose(s, 0.0):
    raise ValueError("Сума елементів w_q = 0. Неможливо нормувати.")

# --------------------- Нормалізація ---------------------
if not np.allclose(col_sum, 1.0):
    W = W / col_sum
    print("Матрицю W автоматично нормовано по стовпцях.")

if not np.isclose(s, 1.0):
    w_q = w_q / s
    print("Вектор ваг w_q автоматично нормовано.")

# --------------------- Агреговані пріоритети ---------------------
w_aggr = W @ w_q   # (n,)

# --------------------- Підготовка пар ---------------------
pairs = np.array(list(itertools.combinations(range(n), 2)))
diffs = W[pairs[:, 0], :] - W[pairs[:, 1], :]
diffs_aggr = w_aggr[pairs[:, 0]] - w_aggr[pairs[:, 1]]

# --------------------- Вивід у консоль ---------------------
print("\n===== Нормована матриця W =====")
print(W)
print("\n===== Нормований вектор ваг w_q =====")
print(w_q)
print("\n===== Агреговані пріоритети моделей =====")
for i, val in enumerate(w_aggr, start=1):
    print(f"Модель {i}: {val:.3f}")

# --------------------- Візуалізація ---------------------
num_pairs = len(pairs)
cols = 2
rows = (num_pairs + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
axes = axes.flatten()

x = np.arange(1, m + 1)

for idx, (i, k) in enumerate(pairs):
    ax = axes[idx]
    ax.bar(x, diffs[idx], color="skyblue", edgecolor="black", label="Різниці w[i,j]-w[k,j]")
    ax.axhline(0, color="black", linewidth=1)

    ax.set_title(f"Модель {i+1} vs {k+1}")
    ax.set_xlabel("Показник j")
    ax.set_ylabel("Різниця пріоритетів")
    ax.set_xticks(x)

    ax.text(
        0.02, 0.95,
        f"Агрегована різниця = {diffs_aggr[idx]:.3f}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9, color="red"
    )

    ax.legend()

plt.tight_layout()
plt.show()
