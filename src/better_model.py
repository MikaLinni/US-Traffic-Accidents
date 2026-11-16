"""Проверка градиентного бустинга на сбалансированном train."""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from src.modeling import LABELS, WEATHER_NUMBERS, WEATHER_CATEGORY, fit_with_measurement
from src.kaggle_experiment import EXTRA_NUMERIC, EXTRA_CATEGORIES, EXTENDED_FEATURES
from src.large_balanced_experiment import balance_train
from src.stratified_experiment import balanced_evaluation


def create_boosting():
    """Признаки готовятся только на train и используются бустингом."""
    numeric = ['hour', 'day_of_week'] + WEATHER_NUMBERS + EXTRA_NUMERIC
    category = ['State', 'Sunrise_Sunset'] + WEATHER_CATEGORY + EXTRA_CATEGORIES
    prep = ColumnTransformer([
        ('numbers', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), category),
    ], sparse_threshold=0)
    classifier = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31,
                                                 learning_rate=0.1, random_state=42,
                                                 early_stopping=False)
    return Pipeline([('prepare', prep), ('model', classifier)])


def evaluate_boosting(train, validation):
    """Метрики на validation с естественным и равным распределением."""
    balanced_train = balance_train(train, target_per_class=50000)
    balanced_val = balanced_evaluation(validation)
    model = create_boosting()
    print('Число строк balanced train:', len(balanced_train), flush=True)
    print('Число уникальных ID:', balanced_train.ID.nunique(), flush=True)
    print('Обучаем Gradient Boosting...', flush=True)
    with threadpool_limits(limits=4):
        seconds, memory = fit_with_measurement(model, balanced_train[EXTENDED_FEATURES], balanced_train.Severity)
        natural_pred = model.predict(validation[EXTENDED_FEATURES])
        balanced_pred = model.predict(balanced_val[EXTENDED_FEATURES])
    scores = {'model':'Gradient Boosting + road / location',
              'train_rows':len(balanced_train),
              'accuracy_natural':accuracy_score(validation.Severity, natural_pred),
              'f1_macro_natural':f1_score(validation.Severity, natural_pred, labels=LABELS, average='macro', zero_division=0),
              'accuracy_equal_classes':accuracy_score(balanced_val.Severity, balanced_pred),
              'f1_macro_equal_classes':f1_score(balanced_val.Severity, balanced_pred, labels=LABELS, average='macro', zero_division=0),
              'train_seconds':seconds, 'peak_process_ram_mib':memory}
    return scores, model, natural_pred, balanced_val, balanced_pred
