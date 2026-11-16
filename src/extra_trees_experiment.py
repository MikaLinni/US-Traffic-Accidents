"""Сравнение Extra Trees с предыдущей моделью на validation."""
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.modeling import FEATURES, LABELS, build_pipeline, fit_with_measurement


def try_extra_trees(train, validation):
    """Обучает два варианта Extra Trees на одних и тех же признаках."""
    weights = {1: 3, 2: 1, 3: 2, 4: 8}
    variants = {
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=18,
                          min_samples_leaf=5, n_jobs=2, random_state=42),
        'Extra Trees с весами': ExtraTreesClassifier(n_estimators=100, max_depth=18,
                          min_samples_leaf=5, class_weight=weights, n_jobs=2, random_state=42),
    }
    rows, fitted, predictions = [], {}, {}
    for name, estimator in variants.items():
        print('Обучаем:', name, flush=True)
        model = build_pipeline(estimator)
        seconds, memory = fit_with_measurement(model, train[FEATURES], train['Severity'])
        predicted = model.predict(validation[FEATURES])
        rows.append({'model': name, 'accuracy': accuracy_score(validation['Severity'], predicted),
                     'f1_macro': f1_score(validation['Severity'], predicted, labels=LABELS,
                                          average='macro', zero_division=0),
                     'train_seconds': seconds, 'peak_process_ram_mib': memory})
        fitted[name], predictions[name] = model, predicted
    return pd.DataFrame(rows), fitted, predictions
