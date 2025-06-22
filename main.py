import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def run_recognition_system():
    """
    Основная функция для запуска системы распознавания рукописных цифр.
    Выполняет загрузку данных, предобработку, обучение моделей
    и оценку их производительности.
    """

    print("--- Запуск системы распознавания ---")

    # --- 1. Загрузка и первичная обработка данных ---
    print("\n1. Загрузка и первичная обработка данных...")
    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"Исходная форма данных X: {X.shape}")
    print(f"Исходная форма меток y: {y.shape}")

    # Визуализация нескольких примеров изображений
    plt.figure(figsize=(12, 5))
    plt.suptitle('Примеры изображений из набора данных (первичная выборка)', y=1.05, fontsize=16)
    for index, (image, label) in enumerate(zip(digits.images[:10], digits.target[:10])):
        plt.subplot(2, 5, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f'Цифра: {label}', fontsize=10)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Нормализация данных: пиксельные значения от 0 до 16 для digits.data, преобразуем в [0, 1]
    X_normalized = X / 16.0
    print(f"Данные нормализованы до диапазона [0, 1]. Пример: {X_normalized[0, :5]}")

    # Разделение данных на обучающую и тестовую выборки
    # test_size=0.2 (20% для теста), random_state=42 для воспроизводимости,
    # stratify=y для сохранения пропорций классов
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Форма обучающей выборки X_train: {X_train.shape}")
    print(f"Форма тестовой выборки X_test: {X_test.shape}")
    print(f"Распределение классов в обучающей выборке: {np.bincount(y_train)}")
    print(f"Распределение классов в тестовой выборке: {np.bincount(y_test)}")

    # --- 2. Подготовка данных и масштабирование признаков ---
    print("\n2. Масштабирование признаков для чувствительных алгоритмов...")
    # Масштабирование признаков для SVM и Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Обучающие данные масштабированы. Среднее: {np.mean(X_train_scaled):.2f}, Стд. откл.: {np.std(X_train_scaled):.2f}")


    # --- 3. Построение и обучение моделей с подбором гиперпараметров ---
    print("\n3. Построение и обучение моделей (Logistic Regression, SVM, Random Forest) с GridSearchCV...")

    # Определяем модели и диапазоны гиперпараметров для Grid Search
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=10000, solver='liblinear', random_state=42),
            'params': {'C': [0.1, 1, 10]} # C: параметр регуляризации
        },
        'SVM (RBF Kernel)': {
            'model': SVC(probability=True, random_state=42),
            'params': {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]} # C: регуляризация, gamma: параметр ядра
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]} # n_estimators: число деревьев, max_depth: глубина
        }
    }

    # Словарь для хранения лучших обученных моделей и их результатов
    best_models = {}

    for name, config in models.items():
        print(f"\n--- Обучение {name} ---")
        classifier = config['model']
        param_grid = config['params']

        # GridSearchCV для поиска оптимальных гиперпараметров с 5-кратной кросс-валидацией
        # n_jobs=-1 использует все доступные ядра CPU
        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

        # Применяем масштабированные данные для LR и SVM, обычные для RF
        if name in ['Logistic Regression', 'SVM (RBF Kernel)']:
            grid_search.fit(X_train_scaled, y_train)
        else:
            grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score_cv = grid_search.best_score_ # Лучшая средняя точность по кросс-валидации

        print(f"Лучшие параметры для {name}: {best_params}")
        print(f"Лучшая средняя точность кросс-валидации для {name}: {best_score_cv:.4f}")

        best_models[name] = {
            'model': best_model,
            'best_params': best_params,
            'best_cv_score': best_score_cv
        }

    # --- 4. Оценка и верификация моделей на тестовой выборке ---
    print("\n4. Оценка моделей на тестовой выборке...")

    for name, data in best_models.items():
        model = data['model']

        # Предсказания на тестовой выборке (с учетом масштабирования)
        if name in ['Logistic Regression', 'SVM (RBF Kernel)']:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)

        print(f"\n--- Результаты для {name} на тестовой выборке ---")
        # Отчет о классификации (precision, recall, f1-score, support для каждого класса)
        print("Отчет о классификации:")
        print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        print("Матрица ошибок:")
        # Визуализация матрицы ошибок
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[str(i) for i in range(10)],
                    yticklabels=[str(i) for i in range(10)],
                    cbar=False) # Отключаем цветовую шкалу
        plt.xlabel('Предсказанные метки', fontsize=12)
        plt.ylabel('Истинные метки', fontsize=12)
        plt.title(f'Матрица ошибок для {name}', fontsize=14)
        plt.show()

        # Записываем точность на тестовой выборке в результаты
        test_accuracy = model.score(X_test_scaled if name in ['Logistic Regression', 'SVM (RBF Kernel)'] else X_test, y_test)
        best_models[name]['test_accuracy'] = test_accuracy

    print("\n--- Сводка финальных результатов ---")
    for name, data in best_models.items():
        print(f"{name}:")
        print(f"  Лучшие гиперпараметры (GridSearch): {data['best_params']}")
        print(f"  Средняя точность на кросс-валидации: {data['best_cv_score']:.4f}")
        print(f"  Точность на тестовой выборке: {data['test_accuracy']:.4f}")
        print("-" * 30)

    print("\n--- Система распознавания успешно выполнена! ---")

if __name__ == "__main__":
    run_recognition_system()
