"""Последние простые эксперименты с дисбалансом и градиентным бустингом."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.modeling import FEATURES, LABELS, WEATHER_NUMBERS, WEATHER_CATEGORY, build_pipeline, fit_with_measurement


def undersample_majority(train, fraction, seed=42):
    """Уменьшаем класс 2 только в train. Остальные классы не меняем."""
    if not 0 < fraction <= 1:
        raise ValueError('Доля должна быть от 0 до 1')
    common = train[train['Severity'] == 2].sample(frac=fraction, random_state=seed)
    rare = train[train['Severity'] != 2]
    return pd.concat([common, rare]).sample(frac=1, random_state=seed)


def make_boosting_pipeline():
    """Небольшой градиентный бустинг на тех же признаках, что и ранее."""
    numeric = ['hour', 'day_of_week'] + WEATHER_NUMBERS
    categorical = ['State', 'Sunrise_Sunset'] + WEATHER_CATEGORY
    preparation = ColumnTransformer([
        ('numbers', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ], sparse_threshold=0)
    return Pipeline([('prepare', preparation),
                     ('model', HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                               learning_rate=0.08, random_state=42))])


def test_balancing_and_boosting(train, validation):
    """Сравниваем варианты на неизменённой validation, test не используем."""
    variants = [
        ('Logistic: 50% класса 2', build_pipeline(LogisticRegression(max_iter=500, random_state=42)), 0.5, False),
        ('Logistic: 25% класса 2', build_pipeline(LogisticRegression(max_iter=500, random_state=42)), 0.25, False),
        ('Forest: 25% класса 2', build_pipeline(RandomForestClassifier(n_estimators=50, max_depth=14,
                                min_samples_leaf=5, n_jobs=2, random_state=42)), 0.25, False),
        ('Boosting: обычный', make_boosting_pipeline(), 1.0, False),
        ('Boosting: веса классов', make_boosting_pipeline(), 1.0, True),
    ]
    results, fitted, predictions = [], {}, {}
    for name, model, fraction, weighted in variants:
        print('Обучаем:', name, flush=True)
        subset = undersample_majority(train, fraction)
        if weighted:
            # Веса применяются только во время обучения, без изменения validation.
            weights = subset['Severity'].map({1: 3, 2: 1, 3: 2, 4: 8}).to_numpy()
            import time
            import psutil
            start = time.perf_counter()
            model.fit(subset[FEATURES], subset['Severity'], model__sample_weight=weights)
            seconds = time.perf_counter() - start
            memory = np.nan  # Отдельное измерение пика здесь не проводим.
        else:
            seconds, memory = fit_with_measurement(model, subset[FEATURES], subset['Severity'])
        predicted = model.predict(validation[FEATURES])
        results.append({'model': name, 'train_rows': len(subset),
                        'accuracy': accuracy_score(validation['Severity'], predicted),
                        'f1_macro': f1_score(validation['Severity'], predicted,
                                             labels=LABELS, average='macro', zero_division=0),
                        'train_seconds': seconds, 'peak_process_ram_mib': memory})
        fitted[name], predictions[name] = model, predicted
    return pd.DataFrame(results), fitted, predictions
