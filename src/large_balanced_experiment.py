"""Обучение на более крупной выборке и балансировка только train."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.modeling import FEATURES, LABELS, build_pipeline, fit_with_measurement


def balance_train(train, target_per_class=50000, seed=42):
    """Сэмплируем одинаковое число на класс; редкие записи повторяются."""
    groups = []
    for level in LABELS:
        subset = train[train['Severity'] == level]
        if subset.empty:
            raise ValueError(f'В train нет класса {level}')
        groups.append(subset.sample(n=target_per_class, replace=len(subset) < target_per_class, random_state=seed))
    return pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)


def compare_large_training(train, validation):
    """Одна исходная validation, одинаковый train для сравнения моделей."""
    balanced = balance_train(train)
    print('До балансировки:\n', train['Severity'].value_counts().sort_index(), flush=True)
    print('После балансировки:\n', balanced['Severity'].value_counts().sort_index(), flush=True)
    print('Уникальных ID после балансировки:', balanced['ID'].nunique(), flush=True)
    options = {
        'Logistic 30% balanced': build_pipeline(LogisticRegression(max_iter=500, random_state=42)),
        'Forest 30% balanced': build_pipeline(RandomForestClassifier(n_estimators=100, max_depth=18,
                                     min_samples_leaf=5, n_jobs=2, random_state=42)),
    }
    rows, models, predictions = [], {}, {}
    for name, model in options.items():
        print('Обучаем:', name, flush=True)
        seconds, memory = fit_with_measurement(model, balanced[FEATURES], balanced['Severity'])
        predicted = model.predict(validation[FEATURES])
        rows.append({'model': name, 'accuracy': accuracy_score(validation['Severity'], predicted),
                     'f1_macro': f1_score(validation['Severity'], predicted, labels=LABELS, average='macro', zero_division=0),
                     'train_seconds': seconds, 'peak_process_ram_mib': memory})
        models[name], predictions[name] = model, predicted
    return pd.DataFrame(rows), models, predictions, balanced


def compare_full_training(train, validation):
    """Обучаем на всех строках 30% с разными весами классов."""
    options = {
        'Logistic 30% равные веса классов': 'balanced',
        'Logistic 30% умеренные веса': {1: 3, 2: 1, 3: 2, 4: 8},
    }
    rows, models, predictions = [], {}, {}
    for name, weights in options.items():
        print('Обучаем на всех строках train:', name, flush=True)
        model = build_pipeline(LogisticRegression(max_iter=500, class_weight=weights, random_state=42))
        seconds, memory = fit_with_measurement(model, train[FEATURES], train['Severity'])
        pred = model.predict(validation[FEATURES])
        rows.append({'model': name, 'train_rows': len(train),
                     'accuracy': accuracy_score(validation['Severity'], pred),
                     'f1_macro': f1_score(validation['Severity'], pred, labels=LABELS, average='macro', zero_division=0),
                     'train_seconds': seconds, 'peak_process_ram_mib': memory})
        models[name], predictions[name] = model, pred
    return pd.DataFrame(rows), models, predictions
