import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

 
# ЕТАП 1: CIRCLES DATASET - ПРЕДСТАВЛЕННЯ ПОЧАТКОВИХ ДАНИХ
 

print("=" * 80)
print("CIRCLES DATASET - БІНАРНА КЛАСИФІКАЦІЯ")
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

 
# ЕТАП 2: РОЗБИТТЯ НА НАВЧАЛЬНИЙ ТА ВАЛІДАЦІЙНИЙ НАБОРИ
 

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_circles, y_circles, test_size=0.3, random_state=42, stratify=y_circles
)

print(f"\nРозбиття даних:")
print(f"  Навчальна вибірка: {len(X_train_c)} прикладів")
print(f"  Тестова вибірка: {len(X_test_c)} прикладів")

 
# ЕТАП 3: ПОБУДОВА МОДЕЛЕЙ КЛАСИФІКАЦІЇ
 

print("\n" + "=" * 80)
print("ПОБУДОВА МОДЕЛЕЙ")
print("=" * 80)

models_circles = {}

# 1. Проста логістична регресія (без регуляризації)
print("\n1. Проста логістична регресія (без регуляризації)...")
models_circles['Simple (no reg)'] = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty=None, max_iter=2000, random_state=42))
])

# 2. Проста логістична регресія з L2 регуляризацією
print("2. Проста логістична регресія з L2...")
models_circles['Simple L2'] = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', C=1.0, max_iter=2000, 
                              solver='lbfgs', random_state=42))
])

# 3. Проста логістична регресія з L1 регуляризацією
print("3. Проста логістична регресія з L1...")
models_circles['Simple L1'] = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l1', C=1.0, max_iter=2000, 
                              solver='liblinear', random_state=42))
])

# 4. Поліноміальна логістична регресія (без регуляризації)
print("4. Поліноміальна логістична регресія (degree=2, без регуляризації)...")
models_circles['Polynomial (no reg)'] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty=None, max_iter=2000, random_state=42))
])

# 5. Поліноміальна з L2 регуляризацією
print("5. Поліноміальна з L2...")
models_circles['Polynomial L2'] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', C=1.0, max_iter=2000, 
                              solver='lbfgs', random_state=42))
])

# 6. Поліноміальна з L1 регуляризацією
print("6. Поліноміальна з L1...")
models_circles['Polynomial L1'] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l1', C=1.0, max_iter=2000, 
                              solver='saga', random_state=42))
])

# Навчання всіх моделей
for name, model in models_circles.items():
    model.fit(X_train_c, y_train_c)
    print(f"  ✓ {name} - навчено")

 
# ЕТАП 4 & 8: ВІЗУАЛІЗАЦІЯ МОДЕЛЕЙ - ГРАНИЦІ РІШЕНЬ
 

print("\n" + "=" * 80)
print("ВІЗУАЛІЗАЦІЯ ГРАНИЦЬ РІШЕНЬ")
print("=" * 80)

fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Circles Dataset - Decision Boundaries', fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, (name, model) in enumerate(models_circles.items()):
    ax = axes[idx]
    
    # Створення сітки
    h = 0.02
    x_min, x_max = X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5
    y_min, y_max = X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Візуалізація
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X_train_c[:, 0], X_train_c[:, 1], c=y_train_c, 
               cmap='RdYlBu', edgecolors='black', s=40, alpha=0.7, label='Train')
    ax.scatter(X_test_c[:, 0], X_test_c[:, 1], c=y_test_c, 
               cmap='RdYlBu', edgecolors='green', s=40, alpha=0.7, 
               linewidths=2, marker='s', label='Test')
    
    train_acc = accuracy_score(y_train_c, model.predict(X_train_c))
    test_acc = accuracy_score(y_test_c, model.predict(X_test_c))
    ax.set_title(f'{name}\nTrain: {train_acc:.3f} | Test: {test_acc:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

 
# ЕТАП 5 & 7: ПРОГНОЗИ ТА АПОСТЕРІОРНІ ЙМОВІРНОСТІ
 

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТИ МОДЕЛЕЙ ТА АПОСТЕРІОРНІ ЙМОВІРНОСТІ")
print("=" * 80)

for name, model in models_circles.items():
    y_pred_train = model.predict(X_train_c)
    y_pred_test = model.predict(X_test_c)
    
    train_acc = accuracy_score(y_train_c, y_pred_train)
    test_acc = accuracy_score(y_test_c, y_pred_test)
    
    # Апостеріорні ймовірності для першого тестового прикладу
    proba = model.predict_proba(X_test_c[:1])[0]
    
    print(f"\n{name}:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Перший тестовий приклад: {X_test_c[0]}")
    print(f"  Істинний клас: {y_test_c[0]}")
    print(f"  Передбачений клас: {model.predict(X_test_c[:1])[0]}")
    print(f"  P(Class 0): {proba[0]:.4f}, P(Class 1): {proba[1]:.4f}")

 
# ЕТАП 6: ОЦІНКА ПЕРЕНАВЧАННЯ
 

print("\n" + "=" * 80)
print("АНАЛІЗ ПЕРЕНАВЧАННЯ")
print("=" * 80)

overfitting_threshold = 0.1

for name, model in models_circles.items():
    train_acc = accuracy_score(y_train_c, model.predict(X_train_c))
    test_acc = accuracy_score(y_test_c, model.predict(X_test_c))
    diff = train_acc - test_acc
    
    status = "⚠️ ПЕРЕНАВЧАННЯ" if diff > overfitting_threshold else "✓ Без перенавчання"
    print(f"\n{name}:")
    print(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}, Різниця: {diff:.4f}")
    print(f"  Статус: {status}")

# Візуалізація перенавчання
fig3, ax = plt.subplots(figsize=(12, 6))
model_names = list(models_circles.keys())
train_accs = [accuracy_score(y_train_c, models_circles[name].predict(X_train_c)) 
              for name in model_names]
test_accs = [accuracy_score(y_test_c, models_circles[name].predict(X_test_c)) 
             for name in model_names]
differences = [t - te for t, te in zip(train_accs, test_accs)]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='orange', alpha=0.7)

ax.axhline(y=overfitting_threshold, color='red', linestyle='--', 
           label='Overfitting Threshold', linewidth=2)
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Circles Dataset - Overfitting Analysis', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

 
# ЕТАП 9: КРИТЕРІЇ ЯКОСТІ КЛАСИФІКАЦІЇ
 

print("\n" + "=" * 80)
print("КРИТЕРІЇ ЯКОСТІ КЛАСИФІКАЦІЇ")
print("=" * 80)

# Словник для збереження метрик
metrics_results = {}

for name, model in models_circles.items():
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    y_pred_train = model.predict(X_train_c)
    y_pred_test = model.predict(X_test_c)
    y_proba_train = model.predict_proba(X_train_c)[:, 1]
    y_proba_test = model.predict_proba(X_test_c)[:, 1]
    
    # Матриця неточностей
    cm_train = confusion_matrix(y_train_c, y_pred_train)
    cm_test = confusion_matrix(y_test_c, y_pred_test)
    
    print("\nМатриця неточностей (Train):")
    print(cm_train)
    print("\nМатриця неточностей (Test):")
    print(cm_test)
    
    # Метрики
    precision_train = precision_score(y_train_c, y_pred_train, average='binary')
    recall_train = recall_score(y_train_c, y_pred_train, average='binary')
    f1_train = f1_score(y_train_c, y_pred_train, average='binary')
    
    precision_test = precision_score(y_test_c, y_pred_test, average='binary')
    recall_test = recall_score(y_test_c, y_pred_test, average='binary')
    f1_test = f1_score(y_test_c, y_pred_test, average='binary')
    
    print(f"\nНавчальна вибірка:")
    print(f"  Precision: {precision_train:.4f}")
    print(f"  Recall:    {recall_train:.4f}")
    print(f"  F1 Score:  {f1_train:.4f}")
    
    print(f"\nТестова вибірка:")
    print(f"  Precision: {precision_test:.4f}")
    print(f"  Recall:    {recall_test:.4f}")
    print(f"  F1 Score:  {f1_test:.4f}")
    
    # ROC-AUC
    try:
        roc_auc_train = roc_auc_score(y_train_c, y_proba_train)
        roc_auc_test = roc_auc_score(y_test_c, y_proba_test)
        print(f"\n  ROC-AUC (Train): {roc_auc_train:.4f}")
        print(f"  ROC-AUC (Test):  {roc_auc_test:.4f}")
    except:
        roc_auc_train = roc_auc_test = None
    
    # Збереження метрик
    metrics_results[name] = {
        'precision_test': precision_test,
        'recall_test': recall_test,
        'f1_test': f1_test,
        'roc_auc_test': roc_auc_test,
        'accuracy_test': accuracy_score(y_test_c, y_pred_test)
    }

 
# ВІЗУАЛІЗАЦІЯ: ROC CURVES
 

fig4, ax = plt.subplots(figsize=(10, 8))
ax.plot([0, 1], [0, 1], 'k--', label='Random model', linewidth=2)

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
for (name, model), color in zip(models_circles.items(), colors):
    y_proba = model.predict_proba(X_test_c)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_c, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', 
            linewidth=2, color=color)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Circles Dataset', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

 
# ВІЗУАЛІЗАЦІЯ: PRECISION-RECALL CURVES
 

fig5, ax = plt.subplots(figsize=(10, 8))

for (name, model), color in zip(models_circles.items(), colors):
    y_proba = model.predict_proba(X_test_c)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test_c, y_proba)
    ax.plot(recall, precision, label=name, linewidth=2, color=color)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Circles Dataset', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

 
# ЕТАП 11: GRID SEARCH ДЛЯ ПІДБОРУ ГІПЕРПАРАМЕТРІВ
 

print("\n" + "=" * 80)
print("GRID SEARCH - ПІДБІР ГІПЕРПАРАМЕТРІВ")
print("=" * 80)

# Grid Search для поліноміальної моделі з L2
print("\nGrid Search для Polynomial L2 моделі...")

param_grid = {
    'poly__degree': [2, 3],
    'lr__C': [0.01, 0.1, 1, 10, 100],
    'lr__solver': ['lbfgs', 'saga']
}

grid_model = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', max_iter=3000, random_state=42))
])

grid_search = GridSearchCV(grid_model, param_grid, cv=5, scoring='accuracy', 
                           n_jobs=-1, verbose=1)
grid_search.fit(X_train_c, y_train_c)

print(f"\nНайкращі параметри: {grid_search.best_params_}")
print(f"Найкраща точність (CV): {grid_search.best_score_:.4f}")
print(f"Точність на тесті: {grid_search.score(X_test_c, y_test_c):.4f}")

# Додавання найкращої моделі до словника
models_circles['Grid Search Best'] = grid_search.best_estimator_

 
# ЕТАП 13: ВПЛИВ РОЗМІРУ НАВЧАЛЬНОЇ ВИБІРКИ
 

print("\n" + "=" * 80)
print("АНАЛІЗ ВПЛИВУ РОЗМІРУ НАВЧАЛЬНОЇ ВИБІРКИ")
print("=" * 80)

# Вибираємо 3 моделі для аналізу
selected_models = {
    'Simple (no reg)': models_circles['Simple (no reg)'],
    'Polynomial L2': models_circles['Polynomial L2'],
    'Grid Search Best': models_circles['Grid Search Best']
}

fig6, axes = plt.subplots(1, 3, figsize=(18, 5))
fig6.suptitle('Learning Curves - Circles Dataset', fontsize=16, fontweight='bold')

train_sizes = np.linspace(0.1, 1.0, 10)

for idx, (name, model) in enumerate(selected_models.items()):
    ax = axes[idx]
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_c, y_train_c, train_sizes=train_sizes,
        cv=5, n_jobs=-1, random_state=42, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
            label='Training score', linewidth=2, markersize=7)
    ax.fill_between(train_sizes_abs, train_mean - train_std, 
                     train_mean + train_std, alpha=0.2, color='blue')
    
    ax.plot(train_sizes_abs, test_mean, 'o-', color='red', 
            label='CV score', linewidth=2, markersize=7)
    ax.fill_between(train_sizes_abs, test_mean - test_std, 
                     test_mean + test_std, alpha=0.2, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nВисновок: Більший розмір навчальної вибірки покращує якість моделі,")
print("особливо для складних моделей (поліноміальних).")

 
# ЕТАП 12: ВИСНОВКИ ДЛЯ CIRCLES DATASET
 

print("\n" + "=" * 80)
print("ВИСНОВКИ ДЛЯ CIRCLES DATASET")
print("=" * 80)

print("\n1. ЯКІСТЬ МОДЕЛЕЙ:")
print("   Поліноміальні моделі значно краще справляються з нелінійно")
print("   розділюваними даними (концентричні кола).")

print("\n2. НАЙКРАЩА МОДЕЛЬ:")
best_model_name = max(metrics_results, key=lambda k: metrics_results[k]['accuracy_test'])
best_metrics = metrics_results[best_model_name]
print(f"   {best_model_name}")
print(f"   Accuracy: {best_metrics['accuracy_test']:.4f}")
print(f"   F1 Score: {best_metrics['f1_test']:.4f}")
print(f"   ROC-AUC: {best_metrics['roc_auc_test']:.4f}")

print("\n3. ПЕРЕНАВЧАННЯ:")
print("   Моделі без регуляризації схильні до перенавчання.")
print("   L2 регуляризація допомагає зменшити перенавчання.")

print("\n4. ГІПЕРПАРАМЕТРИ:")
print(f"   Grid Search знайшов оптимальні параметри:")
print(f"   {grid_search.best_params_}")

 
# IRIS DATASET - ПОВНИЙ АНАЛІЗ
 

print("\n\n" + "=" * 80)
print("IRIS DATASET - БАГАТОКЛАСОВА КЛАСИФІКАЦІЯ")
print("=" * 80)

# Завантаження даних
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# ЕТАП 1: Візуалізація початкових даних
fig7, axes = plt.subplots(1, 2, figsize=(16, 6))
fig7.suptitle('Iris Dataset - Initial Data', fontsize=16, fontweight='bold')

# Графік 1: Перші дві ознаки
ax = axes[0]
for i, target_name in enumerate(iris.target_names):
    mask = y_iris == i
    ax.scatter(X_iris[mask, 0], X_iris[mask, 1], label=target_name, 
              s=70, alpha=0.7, edgecolors='black')
ax.set_xlabel('Sepal length (cm)', fontsize=11)
ax.set_ylabel('Sepal width (cm)', fontsize=11)
ax.set_title('Features 1-2', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Графік 2: Третя та четверта ознаки
ax = axes[1]
for i, target_name in enumerate(iris.target_names):
    mask = y_iris == i
    ax.scatter(X_iris[mask, 2], X_iris[mask, 3], label=target_name, 
              s=70, alpha=0.7, edgecolors='black')
ax.set_xlabel('Petal length (cm)', fontsize=11)
ax.set_ylabel('Petal width (cm)', fontsize=11)
ax.set_title('Features 3-4', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nДані Iris Dataset:")
print(f"  Кількість прикладів: {len(X_iris)}")
print(f"  Кількість ознак: {X_iris.shape[1]}")
print(f"  Кількість класів: {len(iris.target_names)}")
print(f"  Класи: {iris.target_names}")
print(f"  Розподіл класів: {np.bincount(y_iris)}")

# ЕТАП 2: Розбиття даних
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

print(f"\nРозбиття даних:")
print(f"  Навчальна вибірка: {len(X_train_i)} прикладів")
print(f"  Тестова вибірка: {len(X_test_i)} прикладів")

# ЕТАП 3: Побудова моделей
print("\n" + "=" * 80)
print("ПОБУДОВА МОДЕЛЕЙ ДЛЯ IRIS")
print("=" * 80)

models_iris = {}

# 1. Проста логістична (OvR, без регуляризації)
print("\n1. Проста логістична (OvR, без регуляризації)...")
models_iris['Simple OvR (no reg)'] = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty=None, multi_class='ovr', 
                              max_iter=2000, random_state=42))
])

# 2. Проста з L2 (OvR)
print("2. Проста з L2 (OvR)...")
models_iris['Simple OvR L2'] = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', C=1.0, multi_class='ovr',
                              max_iter=2000, random_state=42))
])

# 3. Multinomial (без регуляризації)
print("3. Multinomial (без регуляризації)...")
models_iris['Multinomial (no reg)'] = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty=None, multi_class='multinomial',
                              solver='lbfgs', max_iter=2000, random_state=42))
])

# 4. Multinomial з L2
print("4. Multinomial з L2...")
models_iris['Multinomial L2'] = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', C=1.0, multi_class='multinomial',
                              solver='lbfgs', max_iter=2000, random_state=42))
])

# 5. Поліноміальна (без регуляризації)
print("5. Поліноміальна (без регуляризації)...")
models_iris['Polynomial (no reg)'] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty=None, multi_class='ovr',
                              max_iter=2000, random_state=42))
])

# 6. Поліноміальна з L2
print("6. Поліноміальна з L2...")
models_iris['Polynomial L2'] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', C=1.0, multi_class='ovr',
                              max_iter=2000, random_state=42))
])

# Навчання всіх моделей
for name, model in models_iris.items():
    model.fit(X_train_i, y_train_i)
    print(f"  ✓ {name} - навчено")

 
# ЕТАП 4 & 8: ВІЗУАЛІЗАЦІЯ - ГРАНИЦІ РІШЕНЬ (2D проекція)
 

print("\n" + "=" * 80)
print("ВІЗУАЛІЗАЦІЯ ГРАНИЦЬ РІШЕНЬ ДЛЯ IRIS")
print("=" * 80)

fig8, axes = plt.subplots(2, 3, figsize=(18, 12))
fig8.suptitle('Iris Dataset - Decision Boundaries (first 2 features)', 
              fontsize=16, fontweight='bold')
axes = axes.ravel()

X_iris_2d = X_iris[:, :2]

for idx, (name, model) in enumerate(models_iris.items()):
    ax = axes[idx]
    
    # Створення копії моделі для 2D
    has_poly = 'poly' in model.named_steps
    lr_params = model.named_steps['lr']
    
    try:
        if has_poly:
            model_2d = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(
                    penalty=lr_params.penalty,
                    C=lr_params.C if lr_params.penalty != 'none' else 1.0,
                    multi_class=lr_params.multi_class,
                    solver=lr_params.solver,
                    max_iter=2000,
                    random_state=42
                ))
            ])
        else:
            model_2d = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(
                    penalty=lr_params.penalty,
                    C=lr_params.C if lr_params.penalty != 'none' else 1.0,
                    multi_class=lr_params.multi_class,
                    solver=lr_params.solver,
                    max_iter=2000,
                    random_state=42
                ))
            ])
        
        model_2d.fit(X_iris_2d, y_iris)
        
        # Створення сітки
        h = 0.02
        x_min, x_max = X_iris_2d[:, 0].min() - 1, X_iris_2d[:, 0].max() + 1
        y_min, y_max = X_iris_2d[:, 1].min() - 1, X_iris_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Візуалізація
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        scatter = ax.scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_iris, 
                            cmap='viridis', edgecolors='black', s=40, alpha=0.7)
        
        test_acc = accuracy_score(y_test_i, model.predict(X_test_i))
        ax.set_title(f'{name}\nAccuracy (4 features): {test_acc:.3f}', 
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Sepal length (cm)')
        ax.set_ylabel('Sepal width (cm)')
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

 
# ЕТАП 5 & 7: ПРОГНОЗИ ТА АПОСТЕРІОРНІ ЙМОВІРНОСТІ
 

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТИ МОДЕЛЕЙ ТА АПОСТЕРІОРНІ ЙМОВІРНОСТІ ДЛЯ IRIS")
print("=" * 80)

for name, model in models_iris.items():
    y_pred_train = model.predict(X_train_i)
    y_pred_test = model.predict(X_test_i)
    
    train_acc = accuracy_score(y_train_i, y_pred_train)
    test_acc = accuracy_score(y_test_i, y_pred_test)
    
    # Апостеріорні ймовірності для першого тестового прикладу
    proba = model.predict_proba(X_test_i[:1])[0]
    
    print(f"\n{name}:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Перший тестовий приклад: {X_test_i[0]}")
    print(f"  Істинний клас: {iris.target_names[y_test_i[0]]}")
    print(f"  Передбачений клас: {iris.target_names[model.predict(X_test_i[:1])[0]]}")
    print(f"  Ймовірності: {', '.join([f'{iris.target_names[i]}: {p:.4f}' for i, p in enumerate(proba)])}")

 
# ЕТАП 6: ОЦІНКА ПЕРЕНАВЧАННЯ ДЛЯ IRIS
 

print("\n" + "=" * 80)
print("АНАЛІЗ ПЕРЕНАВЧАННЯ ДЛЯ IRIS")
print("=" * 80)

overfitting_threshold = 0.1

for name, model in models_iris.items():
    train_acc = accuracy_score(y_train_i, model.predict(X_train_i))
    test_acc = accuracy_score(y_test_i, model.predict(X_test_i))
    diff = train_acc - test_acc
    
    status = "⚠️ ПЕРЕНАВЧАННЯ" if diff > overfitting_threshold else "✓ Без перенавчання"
    print(f"\n{name}:")
    print(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}, Різниця: {diff:.4f}")
    print(f"  Статус: {status}")

# Візуалізація перенавчання
fig9, ax = plt.subplots(figsize=(14, 6))
model_names_i = list(models_iris.keys())
train_accs_i = [accuracy_score(y_train_i, models_iris[name].predict(X_train_i)) 
                for name in model_names_i]
test_accs_i = [accuracy_score(y_test_i, models_iris[name].predict(X_test_i)) 
               for name in model_names_i]

x = np.arange(len(model_names_i))
width = 0.35

bars1 = ax.bar(x - width/2, train_accs_i, width, label='Train Accuracy', 
               color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, test_accs_i, width, label='Test Accuracy', 
               color='orange', alpha=0.7)

ax.axhline(y=overfitting_threshold, color='red', linestyle='--', 
           label='Overfitting Threshold', linewidth=2)
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Iris Dataset - Overfitting Analysis', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names_i, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.9, 1.05])
plt.tight_layout()
plt.show()

 
# ЕТАП 9: КРИТЕРІЇ ЯКОСТІ ДЛЯ IRIS
 

print("\n" + "=" * 80)
print("КРИТЕРІЇ ЯКОСТІ КЛАСИФІКАЦІЇ ДЛЯ IRIS")
print("=" * 80)

metrics_results_iris = {}

for name, model in models_iris.items():
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    y_pred_train = model.predict(X_train_i)
    y_pred_test = model.predict(X_test_i)
    y_proba_train = model.predict_proba(X_train_i)
    y_proba_test = model.predict_proba(X_test_i)
    
    # Матриця неточностей
    cm_train = confusion_matrix(y_train_i, y_pred_train)
    cm_test = confusion_matrix(y_test_i, y_pred_test)
    
    print("\nМатриця неточностей (Train):")
    print(cm_train)
    print("\nМатриця неточностей (Test):")
    print(cm_test)
    
    # Метрики (macro average для multi-class)
    precision_train = precision_score(y_train_i, y_pred_train, average='macro')
    recall_train = recall_score(y_train_i, y_pred_train, average='macro')
    f1_train = f1_score(y_train_i, y_pred_train, average='macro')
    
    precision_test = precision_score(y_test_i, y_pred_test, average='macro')
    recall_test = recall_score(y_test_i, y_pred_test, average='macro')
    f1_test = f1_score(y_test_i, y_pred_test, average='macro')
    
    print(f"\nНавчальна вибірка (macro-average):")
    print(f"  Precision: {precision_train:.4f}")
    print(f"  Recall:    {recall_train:.4f}")
    print(f"  F1 Score:  {f1_train:.4f}")
    
    print(f"\nТестова вибірка (macro-average):")
    print(f"  Precision: {precision_test:.4f}")
    print(f"  Recall:    {recall_test:.4f}")
    print(f"  F1 Score:  {f1_test:.4f}")
    
    # ROC-AUC (OvR для multi-class)
    try:
        roc_auc_train = roc_auc_score(y_train_i, y_proba_train, 
                                      multi_class='ovr', average='macro')
        roc_auc_test = roc_auc_score(y_test_i, y_proba_test, 
                                     multi_class='ovr', average='macro')
        print(f"\n  ROC-AUC (Train, OvR macro): {roc_auc_train:.4f}")
        print(f"  ROC-AUC (Test, OvR macro):  {roc_auc_test:.4f}")
    except Exception as e:
        roc_auc_test = None
        print(f"\n  ROC-AUC розрахунок неможливий: {e}")
    
    # Детальний звіт
    print(f"\nДетальний звіт класифікації (Test):")
    print(classification_report(y_test_i, y_pred_test, 
                                target_names=iris.target_names))
    
    # Збереження метрик
    metrics_results_iris[name] = {
        'precision_test': precision_test,
        'recall_test': recall_test,
        'f1_test': f1_test,
        'roc_auc_test': roc_auc_test,
        'accuracy_test': accuracy_score(y_test_i, y_pred_test)
    }

 
# ВІЗУАЛІЗАЦІЯ: ROC CURVES ДЛЯ IRIS (One-vs-Rest)
 

fig10, axes = plt.subplots(1, 3, figsize=(18, 5))
fig10.suptitle('ROC Curves - Iris Dataset (One-vs-Rest)', 
               fontsize=16, fontweight='bold')

colors_models = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

for class_idx in range(3):
    ax = axes[class_idx]
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    
    for (name, model), color in zip(models_iris.items(), colors_models):
        y_proba = model.predict_proba(X_test_i)[:, class_idx]
        y_true_binary = (y_test_i == class_idx).astype(int)
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})', 
                linewidth=2, color=color)
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'Class: {iris.target_names[class_idx]}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

 
# ВІЗУАЛІЗАЦІЯ: CONFUSION MATRICES
 

fig11, axes = plt.subplots(2, 3, figsize=(18, 12))
fig11.suptitle('Confusion Matrices - Iris Dataset (Test Set)', 
               fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, (name, model) in enumerate(models_iris.items()):
    ax = axes[idx]
    
    y_pred = model.predict(X_test_i)
    cm = confusion_matrix(y_test_i, y_pred)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=iris.target_names,
           yticklabels=iris.target_names,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Додавання чисел у клітинки
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    test_acc = accuracy_score(y_test_i, y_pred)
    ax.set_title(f'{name}\nAccuracy: {test_acc:.3f}', 
                 fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

 
# ЕТАП 11: GRID SEARCH ДЛЯ IRIS
 

print("\n" + "=" * 80)
print("GRID SEARCH ДЛЯ IRIS")
print("=" * 80)

print("\nGrid Search для Multinomial моделі...")

param_grid_iris = {
    'lr__C': [0.01, 0.1, 1, 10, 100],
    'lr__solver': ['lbfgs', 'newton-cg', 'sag']
}

grid_model_iris = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', multi_class='multinomial', 
                              max_iter=3000, random_state=42))
])

grid_search_iris = GridSearchCV(grid_model_iris, param_grid_iris, cv=5, 
                                scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_iris.fit(X_train_i, y_train_i)

print(f"\nНайкращі параметри: {grid_search_iris.best_params_}")
print(f"Найкраща точність (CV): {grid_search_iris.best_score_:.4f}")
print(f"Точність на тесті: {grid_search_iris.score(X_test_i, y_test_i):.4f}")

models_iris['Grid Search Best'] = grid_search_iris.best_estimator_

 
# ЕТАП 13: ВПЛИВ РОЗМІРУ НАВЧАЛЬНОЇ ВИБІРКИ ДЛЯ IRIS
 

print("\n" + "=" * 80)
print("АНАЛІЗ ВПЛИВУ РОЗМІРУ НАВЧАЛЬНОЇ ВИБІРКИ ДЛЯ IRIS")
print("=" * 80)

selected_models_iris = {
    'Simple OvR L2': models_iris['Simple OvR L2'],
    'Multinomial L2': models_iris['Multinomial L2'],
    'Polynomial L2': models_iris['Polynomial L2']
}

fig12, axes = plt.subplots(1, 3, figsize=(18, 5))
fig12.suptitle('Learning Curves - Iris Dataset', fontsize=16, fontweight='bold')

train_sizes = np.linspace(0.2, 1.0, 8)

for idx, (name, model) in enumerate(selected_models_iris.items()):
    ax = axes[idx]
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_i, y_train_i, train_sizes=train_sizes,
        cv=5, n_jobs=-1, random_state=42, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
            label='Training score', linewidth=2, markersize=7)
    ax.fill_between(train_sizes_abs, train_mean - train_std, 
                     train_mean + train_std, alpha=0.2, color='blue')
    
    ax.plot(train_sizes_abs, test_mean, 'o-', color='red', 
            label='CV score', linewidth=2, markersize=7)
    ax.fill_between(train_sizes_abs, test_mean - test_std, 
                     test_mean + test_std, alpha=0.2, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.05])

plt.tight_layout()
plt.show()

 
# ЕТАП 12: ВИСНОВКИ ДЛЯ IRIS
 

print("\n" + "=" * 80)
print("ВИСНОВКИ ДЛЯ IRIS DATASET")
print("=" * 80)

print("\n1. ЯКІСТЬ МОДЕЛЕЙ:")
print("   Всі моделі показують високу точність (>95%) на Iris датасеті,")
print("   оскільки дані добре лінійно розділювані.")

print("\n2. НАЙКРАЩА МОДЕЛЬ:")
best_model_iris = max(metrics_results_iris, 
                      key=lambda k: metrics_results_iris[k]['accuracy_test'])
best_metrics_iris = metrics_results_iris[best_model_iris]
print(f"   {best_model_iris}")
print(f"   Accuracy: {best_metrics_iris['accuracy_test']:.4f}")
print(f"   F1 Score: {best_metrics_iris['f1_test']:.4f}")
if best_metrics_iris['roc_auc_test']:
    print(f"   ROC-AUC: {best_metrics_iris['roc_auc_test']:.4f}")

print("\n3. ПЕРЕНАВЧАННЯ:")
print("   Майже всі моделі не схильні до перенавчання завдяки")
print("   простоті датасету та достатній кількості даних.")

print("\n4. MULTINOMIAL vs OvR:")
print("   Multinomial підхід часто показує трохи кращі результати")
print("   для multi-class задач порівняно з One-vs-Rest.")

print("\n5. ГІПЕРПАРАМЕТРИ:")
print(f"   Grid Search знайшов оптимальні параметри:")
print(f"   {grid_search_iris.best_params_}")

 
# ЗАГАЛЬНІ ВИСНОВКИ
 

print("\n" + "=" * 80)
print("ЗАГАЛЬНІ ВИСНОВКИ")
print("=" * 80)

print("\n📊 CIRCLES DATASET (нелінійно розділювані дані):")
print("   ✓ Поліноміальні ознаки необхідні для успішної класифікації")
print("   ✓ Прості лінійні моделі не справляються (~50% accuracy)")
print("   ✓ Поліноміальні моделі досягають >95% accuracy")
print("   ✓ Регуляризація важлива для запобігання перенавчанню")

print("\n📊 IRIS DATASET (лінійно розділювані дані):")
print("   ✓ Всі моделі показують високу якість (>95% accuracy)")
print("   ✓ Поліноміальні ознаки не дають суттєвого покращення")
print("   ✓ Multinomial підхід трохи кращий за OvR")
print("   ✓ Датасет дуже простий, перенавчання майже відсутнє")

print("\n🎯 РЕКОМЕНДАЦІЇ:")
print("   1. Для нелінійних даних використовуйте поліноміальні ознаки")
print("   2. Завжди застосовуйте регуляризацію (L1 або L2)")
print("   3. Використовуйте Grid Search для оптимізації гіперпараметрів")
print("   4. Аналізуйте learning curves для виявлення перенавчання")
print("   5. Для multi-class задач розгляньте multinomial підхід")

print("\n" + "=" * 80)
print("АНАЛІЗ ЗАВЕРШЕНО ✓")
print("=" * 80)