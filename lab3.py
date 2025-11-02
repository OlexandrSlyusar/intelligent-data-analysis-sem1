import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ЛАБОРАТОРНА РОБОТА: НЕЙРОННІ МЕРЕЖІ MLPClassifier")
print("=" * 80)
print("\nМЕТА: Дослідження архітектури нейронних мереж")
print("      - Визначення оптимальної кількості нейронів в одному шарі")
print("      - Дослідження багатошарових архітектур")
print("      - Порівняння з логістичною регресією")

# ==================================================================================
# ЧАСТИНА 1: CIRCLES DATASET - НЕЛІНІЙНА БІНАРНА КЛАСИФІКАЦІЯ
# ==================================================================================

print("\n" + "=" * 80)
print("CIRCLES DATASET - НЕЛІНІЙНА БІНАРНА КЛАСИФІКАЦІЯ")
print("=" * 80)

# Генерація даних
X_circles, y_circles = make_circles(500, factor=0.1, noise=0.1, random_state=42)

# Візуалізація початкових даних
fig1 = plt.figure(figsize=(10, 8))
plt.scatter(X_circles[y_circles == 0, 0], X_circles[y_circles == 0, 1],
           c='blue', label='Class 0', s=50, alpha=0.7, edgecolors='black')
plt.scatter(X_circles[y_circles == 1, 0], X_circles[y_circles == 1, 1],
           c='red', label='Class 1', s=50, alpha=0.7, edgecolors='black')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Circles Dataset - Initial Data', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nДані Circles Dataset:")
print(f"  Кількість прикладів: {len(X_circles)}")
print(f"  Кількість ознак: {X_circles.shape[1]}")
print(f"  Розподіл класів: Class 0: {sum(y_circles==0)}, Class 1: {sum(y_circles==1)}")

# Розбиття даних
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_circles, y_circles, test_size=0.3, random_state=42, stratify=y_circles
)

print(f"\nРозбиття даних:")
print(f"  Навчальна вибірка: {len(X_train_c)} прикладів")
print(f"  Тестова вибірка: {len(X_test_c)} прикладів")

# ==================================================================================
# ДОСЛІДЖЕННЯ 1: ОДНОШАРОВІ МОДЕЛІ З РІЗНОЮ КІЛЬКІСТЮ НЕЙРОНІВ
# ==================================================================================

print("\n" + "=" * 80)
print("ДОСЛІДЖЕННЯ 1: ОДНОШАРОВІ НЕЙРОННІ МЕРЕЖІ")
print("Тестування різної кількості нейронів в одному прихованому шарі")
print("=" * 80)

# Виправлено: Додано більше значень нейронів для кращого аналізу
neuron_counts = [2, 4, 8, 16, 32, 64, 128, 256]
models_single_layer = {}

for n_neurons in neuron_counts:
    print(f"\nНавчання моделі з {n_neurons} нейронами...")
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(n_neurons,),
                             activation='relu',
                             solver='adam',
                             max_iter=2000,
                             random_state=42,
                             early_stopping=True,
                             validation_fraction=0.1))
    ])
    
    model.fit(X_train_c, y_train_c)
    
    train_acc = accuracy_score(y_train_c, model.predict(X_train_c))
    test_acc = accuracy_score(y_test_c, model.predict(X_test_c))
    
    models_single_layer[f'{n_neurons} neurons'] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc
    }
    
    print(f"  ✓ Train Accuracy: {train_acc:.4f}")
    print(f"  ✓ Test Accuracy: {test_acc:.4f}")

# Візуалізація результатів одношарових моделей
fig2, axes = plt.subplots(3, 3, figsize=(18, 15))  # Виправлено: збільшено кількість subplots
fig2.suptitle('Circles Dataset - Single Layer MLP (різна к-сть нейронів)', 
              fontsize=16, fontweight='bold')
axes = axes.ravel()

h = 0.02
x_min, x_max = X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5
y_min, y_max = X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for idx, (name, info) in enumerate(models_single_layer.items()):
    ax = axes[idx]
    model = info['model']
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X_train_c[:, 0], X_train_c[:, 1], c=y_train_c,
              cmap='RdYlBu', edgecolors='black', s=40, alpha=0.7, label='Train')
    ax.scatter(X_test_c[:, 0], X_test_c[:, 1], c=y_test_c,
              cmap='RdYlBu', edgecolors='green', s=40, alpha=0.7,
              linewidths=2, marker='s', label='Test')
    
    ax.set_title(f'{name}\nTrain: {info["train_acc"]:.3f} | Test: {info["test_acc"]:.3f}',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

# Виправлено: Прибираємо зайві subplots
for idx in range(len(models_single_layer), len(axes)):
    fig2.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# Аналіз оптимальної кількості нейронів
fig3, ax = plt.subplots(figsize=(12, 6))
neurons_list = neuron_counts
train_accs = [models_single_layer[f'{n} neurons']['train_acc'] for n in neurons_list]
test_accs = [models_single_layer[f'{n} neurons']['test_acc'] for n in neurons_list]

ax.plot(neurons_list, train_accs, 'o-', linewidth=2, markersize=8, 
        label='Train Accuracy', color='blue')
ax.plot(neurons_list, test_accs, 'o-', linewidth=2, markersize=8, 
        label='Test Accuracy', color='orange')
ax.set_xlabel('Кількість нейронів', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Вплив кількості нейронів на якість моделі (один шар)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("ВИСНОВОК ПРО ОДНОШАРОВІ МОДЕЛІ:")
print("=" * 80)
best_single = max(models_single_layer.items(), 
                  key=lambda x: x[1]['test_acc'])
print(f"\nНайкраща одношарова модель: {best_single[0]}")
print(f"  Test Accuracy: {best_single[1]['test_acc']:.4f}")
print(f"\nОптимальна кількість нейронів: {best_single[0].split()[0]}")
if best_single[1]['test_acc'] < 0.95:
    print("⚠️  Одношарової моделі недостатньо для якісного розв'язання задачі")
    print("   Необхідно досліджувати багатошарові архітектури")
else:
    print("✓  Одношарової моделі достатньо для задовільного розв'язання")

# ==================================================================================
# ДОСЛІДЖЕННЯ 2: БАГАТОШАРОВІ МОДЕЛІ
# ==================================================================================

print("\n" + "=" * 80)
print("ДОСЛІДЖЕННЯ 2: БАГАТОШАРОВІ НЕЙРОННІ МЕРЕЖІ")
print("=" * 80)

# Виправлено: Додано більш збалансовані архітектури
architectures = {
    'Один шар (8)': (8,),
    'Один шар (16)': (16,),
    'Два шари (8, 4)': (8, 4),
    'Два шари (16, 8)': (16, 8),
    'Два шари (32, 16)': (32, 16),
    'Три шари (16, 8, 4)': (16, 8, 4),
    'Три шари (32, 16, 8)': (32, 16, 8),
    'Три шари (64, 32, 16)': (64, 32, 16)
}

models_multilayer = {}

for name, arch in architectures.items():
    print(f"\nНавчання моделі: {name} - {arch}...")
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=arch,
                             activation='relu',
                             solver='adam',
                             max_iter=2000,
                             random_state=42,
                             early_stopping=True,
                             validation_fraction=0.1))
    ])
    
    model.fit(X_train_c, y_train_c)
    
    train_acc = accuracy_score(y_train_c, model.predict(X_train_c))
    test_acc = accuracy_score(y_test_c, model.predict(X_test_c))
    
    models_multilayer[name] = {
        'model': model,
        'architecture': arch,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_layers': len(arch)
    }
    
    print(f"  ✓ Train Accuracy: {train_acc:.4f}")
    print(f"  ✓ Test Accuracy: {test_acc:.4f}")

# Візуалізація багатошарових моделей
fig4, axes = plt.subplots(2, 4, figsize=(20, 10))
fig4.suptitle('Circles Dataset - Multi-layer MLP Architectures', 
              fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, (name, info) in enumerate(models_multilayer.items()):
    ax = axes[idx]
    model = info['model']
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X_train_c[:, 0], X_train_c[:, 1], c=y_train_c,
              cmap='RdYlBu', edgecolors='black', s=30, alpha=0.7)
    ax.scatter(X_test_c[:, 0], X_test_c[:, 1], c=y_test_c,
              cmap='RdYlBu', edgecolors='green', s=30, alpha=0.7,
              linewidths=2, marker='s')
    
    ax.set_title(f'{name}\nTrain: {info["train_acc"]:.3f} | Test: {info["test_acc"]:.3f}',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.grid(True, alpha=0.3)

# Прибираємо зайві осі
for idx in range(len(models_multilayer), len(axes)):
    fig4.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# Порівняння архітектур
fig5, ax = plt.subplots(figsize=(14, 6))
model_names = list(models_multilayer.keys())
train_accs = [models_multilayer[name]['train_acc'] for name in model_names]
test_accs = [models_multilayer[name]['test_acc'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', 
               color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', 
               color='orange', alpha=0.7)

ax.set_xlabel('Архітектура', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Порівняння різних архітектур нейронних мереж', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ==================================================================================
# АНАЛІЗ ПЕРЕНАВЧАННЯ ТА МЕТРИК
# ==================================================================================

print("\n" + "=" * 80)
print("АНАЛІЗ ПЕРЕНАВЧАННЯ")
print("=" * 80)

overfitting_threshold = 0.1
for name, info in models_multilayer.items():
    diff = info['train_acc'] - info['test_acc']
    status = "⚠️ ПЕРЕНАВЧАННЯ" if diff > overfitting_threshold else "✓ Без перенавчання"
    print(f"\n{name}:")
    print(f"  Train: {info['train_acc']:.4f}, Test: {info['test_acc']:.4f}, Різниця: {diff:.4f}")
    print(f"  Статус: {status}")

# Детальні метрики для найкращої моделі
print("\n" + "=" * 80)
print("ДЕТАЛЬНІ МЕТРИКИ ЯКОСТІ")
print("=" * 80)

best_model_info = max(models_multilayer.items(), key=lambda x: x[1]['test_acc'])
best_name, best_info = best_model_info
best_model = best_info['model']

print(f"\nНайкраща модель: {best_name}")
print(f"Архітектура: {best_info['architecture']}")

y_pred_test = best_model.predict(X_test_c)
y_proba_test = best_model.predict_proba(X_test_c)[:, 1]

cm = confusion_matrix(y_test_c, y_pred_test)
print(f"\nМатриця неточностей (Test):")
print(cm)

precision = precision_score(y_test_c, y_pred_test, average='binary')
recall = recall_score(y_test_c, y_pred_test, average='binary')
f1 = f1_score(y_test_c, y_pred_test, average='binary')
roc_auc = roc_auc_score(y_test_c, y_proba_test)

print(f"\nМетрики якості:")
print(f"  Accuracy:  {best_info['test_acc']:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# Апостеріорні ймовірності
if len(X_test_c) > 0:  # Виправлено: перевірка на наявність тестових даних
    proba = best_model.predict_proba(X_test_c[:1])[0]
    print(f"\nАпостеріорні ймовірності для першого тестового прикладу:")
    print(f"  Приклад: {X_test_c[0]}")
    print(f"  Істинний клас: {y_test_c[0]}")
    print(f"  Передбачений клас: {best_model.predict(X_test_c[:1])[0]}")
    print(f"  P(Class 0): {proba[0]:.4f}, P(Class 1): {proba[1]:.4f}")

# ROC крива
fig6, ax = plt.subplots(figsize=(10, 8))
ax.plot([0, 1], [0, 1], 'k--', label='Random model', linewidth=2)

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
for (name, info), color in zip(list(models_multilayer.items())[:8], colors):
    model = info['model']
    y_proba = model.predict_proba(X_test_c)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_c, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', 
            linewidth=2, color=color)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Circles Dataset (MLP)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================================================================
# GRID SEARCH ДЛЯ ОПТИМІЗАЦІЇ
# ==================================================================================

print("\n" + "=" * 80)
print("GRID SEARCH - ПІДБІР ОПТИМАЛЬНИХ ГІПЕРПАРАМЕТРІВ")
print("=" * 80)

# Виправлено: Спрощено параметри для швидшої роботи
param_grid = {
    'mlp__hidden_layer_sizes': [(16,), (32,), (16, 8), (32, 16)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.001, 0.01]
}

grid_model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(solver='adam', max_iter=1000, random_state=42,
                         early_stopping=True, validation_fraction=0.1))
])

print("\nЗапуск Grid Search...")
grid_search = GridSearchCV(grid_model, param_grid, cv=3, scoring='accuracy',
                          n_jobs=-1, verbose=1)
grid_search.fit(X_train_c, y_train_c)

print(f"\nНайкращі параметри: {grid_search.best_params_}")
print(f"Найкраща точність (CV): {grid_search.best_score_:.4f}")
print(f"Точність на тесті: {grid_search.score(X_test_c, y_test_c):.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.pipeline import Pipeline
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# ==================================================================================
# ЧАСТИНА 2: IRIS DATASET - ПОВНЕ ДОСЛІДЖЕННЯ НЕЙРОННИХ МЕРЕЖ
# ==================================================================================

print("\n" + "=" * 80)
print("IRIS DATASET - БАГАТОКЛАСОВА КЛАСИФІКАЦІЯ")
print("ДОСЛІДЖЕННЯ НЕЙРОННИХ МЕРЕЖ MLPClassifier")
print("=" * 80)

# Завантаження даних
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Візуалізація початкових даних
fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle('Iris Dataset - Початкові дані', fontsize=16, fontweight='bold')

colors = ['#e74c3c', '#2ecc71', '#3498db']
for i, target_name in enumerate(iris.target_names):
    mask = y_iris == i
    axes[0].scatter(X_iris[mask, 0], X_iris[mask, 1], label=target_name,
                   s=70, alpha=0.7, edgecolors='black', c=colors[i])
    axes[1].scatter(X_iris[mask, 2], X_iris[mask, 3], label=target_name,
                   s=70, alpha=0.7, edgecolors='black', c=colors[i])

axes[0].set_xlabel('Sepal length (cm)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Sepal width (cm)', fontsize=11, fontweight='bold')
axes[0].set_title('Довжина vs Ширина чашолистка', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Petal length (cm)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Petal width (cm)', fontsize=11, fontweight='bold')
axes[1].set_title('Довжина vs Ширина пелюстки', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nХарактеристики Iris Dataset:")
print(f"  Кількість прикладів: {len(X_iris)}")
print(f"  Кількість ознак: {X_iris.shape[1]}")
print(f"  Кількість класів: {len(iris.target_names)}")
print(f"  Назви класів: {list(iris.target_names)}")
print(f"  Розподіл класів:")
for i, name in enumerate(iris.target_names):
    print(f"    {name}: {sum(y_iris == i)} прикладів")

# Розбиття даних зі стратифікацією
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

print(f"\nРозбиття даних (stratified):")
print(f"  Навчальна вибірка: {len(X_train_i)} прикладів")
for i, name in enumerate(iris.target_names):
    print(f"    {name}: {sum(y_train_i == i)}")
print(f"  Тестова вибірка: {len(X_test_i)} прикладів")
for i, name in enumerate(iris.target_names):
    print(f"    {name}: {sum(y_test_i == i)}")

# ==================================================================================
# ЕТАП 1: ОДНОШАРОВІ МОДЕЛІ - ДИНАМІЧНЕ ДОДАВАННЯ НЕЙРОНІВ
# ==================================================================================

print("\n" + "=" * 80)
print("ЕТАП 1: ОДНОШАРОВІ МОДЕЛІ")
print("Динамічне додавання нейронів до прихованого шару")
print("=" * 80)

# Тестуємо різну кількість нейронів від 2 до 64
neuron_counts = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
single_layer_models = {}

print("\nТестування різної кількості нейронів в одному шарі...")

for n_neurons in neuron_counts:
    print(f"\nМодель з {n_neurons} нейронами:")
    start_time = time.time()
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(n_neurons,),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.2
        ))
    ])
    
    # Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_i, y_train_i, cv=cv, scoring='accuracy')
    
    # Навчання моделі
    model.fit(X_train_i, y_train_i)
    train_time = time.time() - start_time
    
    # Оцінка
    train_acc = accuracy_score(y_train_i, model.predict(X_train_i))
    test_acc = accuracy_score(y_test_i, model.predict(X_test_i))
    
    # Збереження результатів
    single_layer_models[n_neurons] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'time': train_time,
        'n_iter': len(model.named_steps['mlp'].loss_curve_),
        'final_loss': model.named_steps['mlp'].loss_curve_[-1]
    }
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  CV Accuracy:    {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  Час навчання:   {train_time:.2f}s")
    print(f"  Ітерацій:       {len(model.named_steps['mlp'].loss_curve_)}")

# Візуалізація залежності accuracy від кількості нейронів
fig2, ax = plt.subplots(figsize=(14, 7))

neurons = list(single_layer_models.keys())
train_accs = [single_layer_models[n]['train_acc'] for n in neurons]
test_accs = [single_layer_models[n]['test_acc'] for n in neurons]
cv_means = [single_layer_models[n]['cv_mean'] for n in neurons]
cv_stds = [single_layer_models[n]['cv_std'] for n in neurons]

ax.plot(neurons, train_accs, 'o-', linewidth=2.5, markersize=8, 
        label='Train Accuracy', color='#3498db')
ax.plot(neurons, test_accs, 's-', linewidth=2.5, markersize=8, 
        label='Test Accuracy', color='#e74c3c')
ax.plot(neurons, cv_means, 'd-', linewidth=2.5, markersize=8,
        label='CV Accuracy', color='#2ecc71')
ax.fill_between(neurons, 
                np.array(cv_means) - np.array(cv_stds),
                np.array(cv_means) + np.array(cv_stds),
                alpha=0.2, color='#2ecc71')

# Додавання значень для тестової точності
for n, acc in zip(neurons, test_accs):
    ax.text(n, acc + 0.01, f'{acc:.3f}', 
           ha='center', fontsize=8, fontweight='bold')

ax.set_xlabel('Кількість нейронів', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Вплив кількості нейронів на якість моделі (одношарова мережа)', 
             fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_ylim([0.85, 1.05])
plt.tight_layout()
plt.show()

# Аналіз результатів одношарових моделей
print("\n" + "=" * 80)
print("АНАЛІЗ ОДНОШАРОВИХ МОДЕЛЕЙ")
print("=" * 80)

best_single = max(single_layer_models.items(), key=lambda x: x[1]['test_acc'])
print(f"\nНайкраща одношарова модель: {best_single[0]} нейронів")
print(f"  Train Accuracy: {best_single[1]['train_acc']:.4f}")
print(f"  Test Accuracy:  {best_single[1]['test_acc']:.4f}")
print(f"  CV Accuracy:    {best_single[1]['cv_mean']:.4f} (±{best_single[1]['cv_std']:.4f})")
print(f"  Час навчання:   {best_single[1]['time']:.2f}s")

# Визначення достатності одношарової моделі
threshold_accuracy = 0.95
if best_single[1]['test_acc'] >= threshold_accuracy:
    print(f"\n✓ ВИСНОВОК: Одношарової моделі з {best_single[0]} нейронами ДОСТАТНЬО")
    print(f"  Досягнуто {best_single[1]['test_acc']:.4f} accuracy (поріг: {threshold_accuracy})")
else:
    print(f"\n⚠️  ВИСНОВОК: Одношарової моделі НЕДОСТАТНЬО")
    print(f"  Досягнуто лише {best_single[1]['test_acc']:.4f} accuracy (поріг: {threshold_accuracy})")
    print(f"  Необхідно тестувати багатошарові архітектури")

# ==================================================================================
# ЕТАП 2: БАГАТОШАРОВІ АРХІТЕКТУРИ
# ==================================================================================

print("\n" + "=" * 80)
print("ЕТАП 2: БАГАТОШАРОВІ НЕЙРОННІ МЕРЕЖІ")
print("Систематичне тестування різних архітектур")
print("=" * 80)

# Визначаємо архітектури для тестування
multilayer_architectures = {
    # Два шари - різні комбінації
    'Два шари (8, 4)': (8, 4),
    'Два шари (10, 5)': (10, 5),
    'Два шари (12, 6)': (12, 6),
    'Два шари (16, 8)': (16, 8),
    'Два шари (20, 10)': (20, 10),
    'Два шари (24, 12)': (24, 12),
    
    # Три шари - пірамідальні
    'Три шари (12, 8, 4)': (12, 8, 4),
    'Три шари (16, 12, 8)': (16, 12, 8),
    'Три шари (20, 15, 10)': (20, 15, 10),
    'Три шари (24, 16, 8)': (24, 16, 8),
    
    # Чотири шари
    'Чотири шари (16, 12, 8, 4)': (16, 12, 8, 4),
    'Чотири шари (20, 16, 12, 8)': (20, 16, 12, 8)
}

multilayer_models = {}

print("\nНавчання багатошарових моделей...")

for name, arch in multilayer_architectures.items():
    print(f"\n{name}:")
    start_time = time.time()
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=arch,
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.2
        ))
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_i, y_train_i, cv=cv, scoring='accuracy')
    
    # Навчання
    model.fit(X_train_i, y_train_i)
    train_time = time.time() - start_time
    
    # Оцінка
    train_acc = accuracy_score(y_train_i, model.predict(X_train_i))
    test_acc = accuracy_score(y_test_i, model.predict(X_test_i))
    
    multilayer_models[name] = {
        'model': model,
        'architecture': arch,
        'n_layers': len(arch),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'time': train_time,
        'n_iter': len(model.named_steps['mlp'].loss_curve_),
        'predictions': model.predict(X_test_i)
    }
    
    print(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}, CV: {cv_scores.mean():.4f}")

# Візуалізація багатошарових моделей
fig3, ax = plt.subplots(figsize=(16, 8))

model_names = list(multilayer_models.keys())
train_accs_multi = [multilayer_models[name]['train_acc'] for name in model_names]
test_accs_multi = [multilayer_models[name]['test_acc'] for name in model_names]
cv_means_multi = [multilayer_models[name]['cv_mean'] for name in model_names]

x = np.arange(len(model_names))
width = 0.25

bars1 = ax.bar(x - width, train_accs_multi, width, label='Train Accuracy',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, test_accs_multi, width, label='Test Accuracy',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, cv_means_multi, width, label='CV Accuracy',
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

# Додавання значень
for i, (t, te, c) in enumerate(zip(train_accs_multi, test_accs_multi, cv_means_multi)):
    ax.text(i - width, t + 0.005, f'{t:.3f}', ha='center', va='bottom', 
            fontsize=7, fontweight='bold')
    ax.text(i, te + 0.005, f'{te:.3f}', ha='center', va='bottom', 
            fontsize=7, fontweight='bold')
    ax.text(i + width, c + 0.005, f'{c:.3f}', ha='center', va='bottom', 
            fontsize=7, fontweight='bold')

ax.set_xlabel('Архітектура', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Порівняння багатошарових архітектур для Iris Dataset', 
             fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.4, axis='y', linestyle='--')
ax.set_ylim([0.85, 1.05])
plt.tight_layout()
plt.show()

# ==================================================================================
# ПОРІВНЯННЯ ОДНОШАРОВИХ ТА БАГАТОШАРОВИХ МОДЕЛЕЙ
# ==================================================================================

print("\n" + "=" * 80)
print("ПОРІВНЯННЯ: ОДНОШАРОВІ vs БАГАТОШАРОВІ МОДЕЛІ")
print("=" * 80)

best_multilayer = max(multilayer_models.items(), key=lambda x: x[1]['test_acc'])

print(f"\nОдношарова модель:")
print(f"  Найкраща: {best_single[0]} нейронів")
print(f"  Test Accuracy: {best_single[1]['test_acc']:.4f}")
print(f"  CV Accuracy: {best_single[1]['cv_mean']:.4f}")

print(f"\nБагатошарова модель:")
print(f"  Найкраща: {best_multilayer[0]}")
print(f"  Архітектура: {best_multilayer[1]['architecture']}")
print(f"  Test Accuracy: {best_multilayer[1]['test_acc']:.4f}")
print(f"  CV Accuracy: {best_multilayer[1]['cv_mean']:.4f}")

improvement = (best_multilayer[1]['test_acc'] - best_single[1]['test_acc']) * 100
if improvement > 0:
    print(f"\n✓ Багатошарова модель кращаза на {improvement:.2f}%")
else:
    print(f"\n→ Одношарової моделі достатньо (різниця: {abs(improvement):.2f}%)")

# ==================================================================================
# CONFUSION MATRICES ДЛЯ КРАЩИХ МОДЕЛЕЙ
# ==================================================================================

print("\n" + "=" * 80)
print("МАТРИЦІ НЕТОЧНОСТЕЙ")
print("=" * 80)

# Вибираємо топ-6 моделей
all_models = {**{f'Один шар ({k})': v for k, v in single_layer_models.items()}, 
              **multilayer_models}
top_models = sorted(all_models.items(), key=lambda x: x[1]['test_acc'], reverse=True)[:6]

fig4, axes = plt.subplots(2, 3, figsize=(18, 12))
fig4.suptitle('Confusion Matrices - Топ-6 моделей для Iris', 
              fontsize=18, fontweight='bold')
axes = axes.ravel()

for idx, (name, info) in enumerate(top_models):
    ax = axes[idx]
    
    # Отримуємо передбачення
    if 'predictions' in info:
        y_pred = info['predictions']
    else:
        y_pred = info['model'].predict(X_test_i)
    
    cm = confusion_matrix(y_test_i, y_pred)
    
    # Теплова карта
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Налаштування осей
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=iris.target_names,
           yticklabels=iris.target_names)
    
    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_ylabel('True', fontsize=10, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Числа в клітинках
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    # Заголовок
    test_acc = info['test_acc']
    cv_acc = info['cv_mean']
    ax.set_title(f'{name}\nTest: {test_acc:.4f} | CV: {cv_acc:.4f}',
                fontsize=10, fontweight='bold', pad=8)

plt.tight_layout()
plt.show()

# ==================================================================================
# ДЕТАЛЬНІ МЕТРИКИ ДЛЯ НАЙКРАЩОЇ МОДЕЛІ
# ==================================================================================

print("\n" + "=" * 80)
print("ДЕТАЛЬНИЙ АНАЛІЗ НАЙКРАЩОЇ МОДЕЛІ")
print("=" * 80)

best_overall = max(all_models.items(), key=lambda x: x[1]['test_acc'])
best_name_overall, best_info_overall = best_overall

print(f"\nНайкраща модель: {best_name_overall}")
if 'architecture' in best_info_overall:
    print(f"Архітектура: {best_info_overall['architecture']}")
print(f"Test Accuracy: {best_info_overall['test_acc']:.4f}")
print(f"CV Accuracy: {best_info_overall['cv_mean']:.4f}")
print(f"Час навчання: {best_info_overall['time']:.2f}s")

# Отримуємо передбачення
if 'predictions' in best_info_overall:
    y_pred_best = best_info_overall['predictions']
else:
    y_pred_best = best_info_overall['model'].predict(X_test_i)

# Macro/Micro метрики
precision_macro = precision_score(y_test_i, y_pred_best, average='macro')
recall_macro = recall_score(y_test_i, y_pred_best, average='macro')
f1_macro = f1_score(y_test_i, y_pred_best, average='macro')

precision_micro = precision_score(y_test_i, y_pred_best, average='micro')
recall_micro = recall_score(y_test_i, y_pred_best, average='micro')
f1_micro = f1_score(y_test_i, y_pred_best, average='micro')

print(f"\nMacro-averaged метрики:")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall:    {recall_macro:.4f}")
print(f"  F1-Score:  {f1_macro:.4f}")

print(f"\nMicro-averaged метрики:")
print(f"  Precision: {precision_micro:.4f}")
print(f"  Recall:    {recall_micro:.4f}")
print(f"  F1-Score:  {f1_micro:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test_i, y_pred_best, target_names=iris.target_names))

# Велика матриця неточностей для найкращої моделі
fig5, ax = plt.subplots(figsize=(10, 8))
cm_best = confusion_matrix(y_test_i, y_pred_best)

im = ax.imshow(cm_best, interpolation='nearest', cmap='YlGnBu')
cbar = ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm_best.shape[1]),
       yticks=np.arange(cm_best.shape[0]),
       xticklabels=iris.target_names,
       yticklabels=iris.target_names)

ax.set_xlabel('Predicted label', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('True label', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(f'Confusion Matrix - {best_name_overall}\nTest Accuracy: {best_info_overall["test_acc"]:.4f}',
             fontsize=16, fontweight='bold', pad=15)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)

# Числа з процентами
thresh = cm_best.max() / 2.
for i in range(cm_best.shape[0]):
    for j in range(cm_best.shape[1]):
        percentage = cm_best[i, j] / cm_best[i].sum() * 100
        text = f'{cm_best[i, j]}\n({percentage:.1f}%)'
        ax.text(j, i, text, ha="center", va="center",
               color="white" if cm_best[i, j] > thresh else "black",
               fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ==================================================================================
# ROC CURVES (One-vs-Rest)
# ==================================================================================

print("\n" + "=" * 80)
print("ROC CURVES")
print("=" * 80)

best_model = best_info_overall['model']
y_proba = best_model.predict_proba(X_test_i)

fig6, axes = plt.subplots(1, 3, figsize=(18, 6))
fig6.suptitle(f'ROC Curves - {best_name_overall}', fontsize=16, fontweight='bold')

colors_roc = ['#e74c3c', '#2ecc71', '#3498db']

for class_idx in range(3):
    ax = axes[class_idx]
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.50)')
    
    y_true_binary = (y_test_i == class_idx).astype(int)
    y_proba_class = y_proba[:, class_idx]
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_proba_class)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, linewidth=3, color=colors_roc[class_idx],
           label=f'{iris.target_names[class_idx]} (AUC={roc_auc:.3f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Class: {iris.target_names[class_idx]}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.4, linestyle='--')

plt.tight_layout()
plt.show()

# ==================================================================================
# GRID SEARCH ДЛЯ ОПТИМІЗАЦІЇ
# ==================================================================================

print("\n" + "=" * 80)
print("GRID SEARCH - ОПТИМАЛЬНІ ГІПЕРПАРАМЕТРИ")
print("=" * 80)

# Базуємось на результатах попередніх експериментів
param_grid = {
    'mlp__hidden_layer_sizes': [(10,), (12,), (16,), (12, 6), (16, 8), (20, 10)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.001, 0.01, 0.1],
    'mlp__learning_rate_init': [0.0001, 0.001, 0.01]
}

grid_model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(solver='adam', max_iter=2000, random_state=42, early_stopping=True))
])

print("\nЗапуск Grid Search...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(grid_model, param_grid, cv=cv, scoring='accuracy',
                          n_jobs=-1, verbose=1)

start_time = time.time()
grid_search.fit(X_train_i, y_train_i)
grid_time = time.time() - start_time

print(f"\nРезультати Grid Search:")
print(f"  Найкращі параметри: {grid_search.best_params_}")
print(f"  Найкраща CV точність: {grid_search.best_score_:.4f}")
print(f"  Test точність: {grid_search.score(X_test_i, y_test_i):.4f}")
print(f"  Час виконання: {grid_time:.2f}s")


