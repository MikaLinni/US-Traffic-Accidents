"""Эксперимент: стратифицированное разбиение и баланс классов."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.modeling import FEATURES, LABELS, build_pipeline, fit_with_measurement
from src.large_balanced_experiment import balance_train


def stratified_split(sample):
    """60/20/20 со сходными долями Severity; test оставляем отдельно."""
    if sample['ID'].duplicated().any():
        raise ValueError('Есть повторяющиеся ID — сначала нужно проверить дубликаты.')
    train, rest = train_test_split(sample, test_size=0.4, random_state=42,
                                   stratify=sample['Severity'])
    validation, test = train_test_split(rest, test_size=0.5, random_state=42,
                                        stratify=rest['Severity'])
    assert set(train.ID).isdisjoint(validation.ID)
    assert set(train.ID).isdisjoint(test.ID)
    assert set(validation.ID).isdisjoint(test.ID)
    return train, validation, test


def balanced_evaluation(validation):
    """Равная оценка без повторов: берём столько строк, сколько у редчайшего класса."""
    counts = validation['Severity'].value_counts().reindex(LABELS, fill_value=0)
    count = int(counts.min())
    if count == 0:
        raise ValueError('В validation отсутствует один из классов')
    return (validation.groupby('Severity', group_keys=False)
            .sample(n=count, random_state=42)
            .sample(frac=1, random_state=42))


def evaluate_stratified(train, validation):
    """Подбираем на validation, test не передаётся в эту функцию."""
    balanced_train = balance_train(train, target_per_class=50000)
    balanced_val = balanced_evaluation(validation)
    print('Состав train после балансировки:', balanced_train.Severity.value_counts().sort_index().to_dict(), flush=True)
    print('Уникальные ID в train после балансировки:', balanced_train.ID.nunique(), flush=True)
    print('Состав равной validation:', balanced_val.Severity.value_counts().sort_index().to_dict(), flush=True)
    models = {
        'Logistic — balanced train': build_pipeline(LogisticRegression(max_iter=500, random_state=42)),
        'Random Forest — balanced train': build_pipeline(RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=5, n_jobs=2, random_state=42)),
        'Logistic — all train, moderate weights': build_pipeline(LogisticRegression(max_iter=500, random_state=42, class_weight={1:3, 2:1, 3:2, 4:8})),
    }
    rows, fitted, predictions = [], {}, {}
    for name, model in models.items():
        subset = train if 'all train' in name else balanced_train
        print('Обучаем:', name, flush=True)
        seconds, memory = fit_with_measurement(model, subset[FEATURES], subset['Severity'])
        natural_prediction = model.predict(validation[FEATURES])
        balanced_prediction = model.predict(balanced_val[FEATURES])
        rows.append({'model': name, 'train_rows': len(subset),
                     'accuracy_natural': accuracy_score(validation.Severity, natural_prediction),
                     'f1_macro_natural': f1_score(validation.Severity, natural_prediction, labels=LABELS, average='macro', zero_division=0),
                     'accuracy_equal_classes': accuracy_score(balanced_val.Severity, balanced_prediction),
                     'f1_macro_equal_classes': f1_score(balanced_val.Severity, balanced_prediction, labels=LABELS, average='macro', zero_division=0),
                     'train_seconds': seconds, 'peak_process_ram_mib': memory})
        fitted[name] = model
        predictions[name] = (natural_prediction, balanced_prediction)
    return pd.DataFrame(rows), fitted, predictions, balanced_val, balanced_train
