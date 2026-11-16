"""Подготовка выборки и сравнение моделей без утечки по времени."""
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.quality_hypotheses import weather_group


BASE_COLUMNS = ['State', 'hour', 'day_of_week', 'Sunrise_Sunset']
WEATHER_NUMBERS = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)']
WEATHER_CATEGORY = ['weather_group']
FEATURES = BASE_COLUMNS + WEATHER_NUMBERS + WEATHER_CATEGORY
LABELS = [1, 2, 3, 4]


def create_model_sample(folder, sample_percent=3, validation_percent=None, train_percent=None, extra_columns=None):
    """Берём воспроизводимую долю по хешу ID и готовим признаки."""
    files = sorted(Path(folder).glob('year=*/part-*.parquet'))
    if not files:
        raise FileNotFoundError('Файлы Parquet не найдены')
    if not 0 < sample_percent <= 100:
        raise ValueError('sample_percent должен быть от 0 до 100')
    if validation_percent is not None and not 0 < validation_percent <= 100:
        raise ValueError('validation_percent должен быть от 0 до 100')
    if train_percent is not None and not 0 < train_percent <= 100:
        raise ValueError('train_percent должен быть от 0 до 100')
    columns = ['ID', 'Severity', 'Start_Time', 'State', 'Sunrise_Sunset',
               'Weather_Timestamp', 'Weather_Condition'] + WEATHER_NUMBERS
    if extra_columns:
        columns += [name for name in extra_columns if name not in columns]
    frames = []
    for file in files:
        data = pd.read_parquet(file, columns=columns)
        # ID нужен только для отбора и проверки пересечений, в модель он не попадёт.
        valid_id = data['ID'].notna()
        hashes = pd.util.hash_pandas_object(data['ID'], index=False)
        selected = hashes % 10000 < sample_percent * 100
        if train_percent is not None:
            years = pd.to_numeric(data['Start_Time'].str.slice(0, 4), errors='coerce')
            is_train = years < 2021
            selected = selected.where(~is_train, hashes % 10000 < train_percent * 100)
        if validation_percent is not None:
            is_validation = data['Start_Time'].str.startswith('2021-', na=False)
            selected = selected.where(~is_validation, hashes % 10000 < validation_percent * 100)
        data = data.loc[valid_id & selected].copy()
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    data['Severity'] = pd.to_numeric(data['Severity'], errors='coerce')
    data['start'] = pd.to_datetime(data['Start_Time'], format='mixed', errors='coerce')
    data['weather_time'] = pd.to_datetime(data['Weather_Timestamp'], format='mixed', errors='coerce')
    data = data.loc[data['Severity'].isin(LABELS) & data['start'].notna()].copy()
    data['Severity'] = data['Severity'].astype(int)
    data['hour'] = data['start'].dt.hour
    data['day_of_week'] = data['start'].dt.dayofweek
    # Погода допустима, только если измерена не позже ДТП и не старше 3 часов.
    age = data['start'] - data['weather_time']
    timely_weather = age.between(pd.Timedelta(0), pd.Timedelta(hours=3))
    data['weather_available'] = timely_weather
    for col in WEATHER_NUMBERS:
        data[col] = pd.to_numeric(data[col], errors='coerce').where(timely_weather)
    # Группировка погоды задана в предыдущем разделе ноутбука.
    data['weather_group'] = data['Weather_Condition'].map(weather_group).where(timely_weather, 'Unknown')
    data['Sunrise_Sunset'] = data['Sunrise_Sunset'].fillna('Unknown')
    data['State'] = data['State'].fillna('Unknown')
    return data


def split_by_time(data):
    """Train: до 2021, validation: 2021, test: 2022–2023."""
    train = data[data['start'].dt.year < 2021].copy()
    validation = data[data['start'].dt.year == 2021].copy()
    test = data[data['start'].dt.year >= 2022].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError('Нужны данные всех трёх периодов. Используйте полный набор или другую выборку.')
    assert set(train['ID']).isdisjoint(validation['ID'])
    assert set(train['ID']).isdisjoint(test['ID'])
    assert set(validation['ID']).isdisjoint(test['ID'])
    return train, validation, test


def build_pipeline(model):
    """Заполнение пропусков и кодирование обучаются только на train."""
    numeric = ['hour', 'day_of_week'] + WEATHER_NUMBERS
    categorical = ['State', 'Sunrise_Sunset'] + WEATHER_CATEGORY
    numeric_steps = Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
                              ('scale', StandardScaler())])
    categories = OneHotEncoder(handle_unknown='ignore')
    preparation = ColumnTransformer([('numbers', numeric_steps, numeric),
                                     ('categories', categories, categorical)])
    return Pipeline([('prepare', preparation), ('model', model)])


def fit_with_measurement(model, x, y):
    """Измеряем длительность и примерный пик RAM всего текущего процесса."""
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(0.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=monitor, daemon=True)
    started = time.perf_counter()
    thread.start()
    try:
        model.fit(x, y)
    finally:
        stop.set()
        thread.join()
    return time.perf_counter() - started, peak[0] / 1024**2


def compare_models(train, validation):
    """Сравниваем только на validation; test остаётся для финальной оценки."""
    models = {
        'Dummy': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': build_pipeline(LogisticRegression(max_iter=500, random_state=42)),
        'Random Forest': build_pipeline(RandomForestClassifier(n_estimators=50, max_depth=14,
                                      min_samples_leaf=5, n_jobs=2, random_state=42)),
    }
    rows, predictions = [], {}
    for name, model in models.items():
        print('Обучаем:', name, flush=True)
        seconds, memory = fit_with_measurement(model, train[FEATURES], train['Severity'])
        predicted = model.predict(validation[FEATURES])
        predictions[name] = predicted
        rows.append({'model': name, 'accuracy': accuracy_score(validation['Severity'], predicted),
                     'f1_macro': f1_score(validation['Severity'], predicted, labels=LABELS, average='macro', zero_division=0),
                     'train_seconds': seconds, 'peak_process_ram_mib': memory})
    return pd.DataFrame(rows), models, predictions


def improve_models(train, validation):
    """Сравнение моделей с разным учётом редких классов на validation."""
    candidates = {
        'Logistic Regression': build_pipeline(LogisticRegression(max_iter=500, random_state=42)),
        'Logistic Regression balanced': build_pipeline(LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)),
        'Logistic Regression moderate': build_pipeline(LogisticRegression(max_iter=500, class_weight={1: 3, 2: 1, 3: 1.5, 4: 2}, random_state=42)),
        'Logistic Regression weight 3/4 = 2/4': build_pipeline(LogisticRegression(max_iter=500, class_weight={1: 3, 2: 1, 3: 2, 4: 4}, random_state=42)),
        'Logistic Regression weight 3/4 = 2/8': build_pipeline(LogisticRegression(max_iter=500, class_weight={1: 3, 2: 1, 3: 2, 4: 8}, random_state=42)),
        'Logistic Regression weight 3/4 = 3/8': build_pipeline(LogisticRegression(max_iter=500, class_weight={1: 3, 2: 1, 3: 3, 4: 8}, random_state=42)),
        'Random Forest balanced': build_pipeline(RandomForestClassifier(n_estimators=50, max_depth=14,
                  min_samples_leaf=5, class_weight='balanced', n_jobs=2, random_state=42)),
    }
    scores, fitted, predictions = [], {}, {}
    for name, model in candidates.items():
        print('Проверяем:', name, flush=True)
        seconds, memory = fit_with_measurement(model, train[FEATURES], train['Severity'])
        predicted = model.predict(validation[FEATURES])
        scores.append({'model': name,
                       'accuracy': accuracy_score(validation['Severity'], predicted),
                       'f1_macro': f1_score(validation['Severity'], predicted, labels=LABELS, average='macro', zero_division=0),
                       'train_seconds': seconds, 'peak_process_ram_mib': memory})
        fitted[name] = model
        predictions[name] = predicted
    return pd.DataFrame(scores), fitted, predictions
