"""Проверка идей из примеров Kaggle на прежнем временном разбиении."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.modeling import FEATURES, LABELS, WEATHER_NUMBERS, WEATHER_CATEGORY, fit_with_measurement


EXTRA_NUMERIC = ['Start_Lat', 'Start_Lng', 'month']
EXTRA_CATEGORIES = ['Crossing', 'Junction', 'Traffic_Signal', 'Stop']
EXTENDED_FEATURES = FEATURES + EXTRA_NUMERIC + EXTRA_CATEGORIES


def add_kaggle_features(sample, folder):
    """Добавляем данные только по ID уже выбранных записей."""
    ids = set(sample['ID'])
    columns = ['ID', 'Start_Lat', 'Start_Lng'] + EXTRA_CATEGORIES
    pieces = []
    for file in sorted(Path(folder).glob('year=*/part-*.parquet')):
        data = pd.read_parquet(file, columns=columns)
        pieces.append(data[data['ID'].isin(ids)])
    extra = pd.concat(pieces, ignore_index=True)
    if extra['ID'].duplicated().any():
        raise ValueError('Повтор ID — невозможно однозначно соединить данные.')
    data = sample.merge(extra, on='ID', how='left', validate='one_to_one')
    if len(data) != len(sample):
        raise ValueError('При соединении изменилось число строк')
    data['month'] = data['start'].dt.month
    for name in ['Start_Lat', 'Start_Lng']:
        data[name] = pd.to_numeric(data[name], errors='coerce')
    for name in EXTRA_CATEGORIES:
        data[name] = data[name].fillna('Unknown').astype(str)
    return data


def make_extended_pipeline(model):
    """Заполнение пропусков и подготовка новых признаков только на train."""
    numbers = ['hour', 'day_of_week'] + WEATHER_NUMBERS + EXTRA_NUMERIC
    categories = ['State', 'Sunrise_Sunset'] + WEATHER_CATEGORY + EXTRA_CATEGORIES
    numeric_steps = Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
                              ('scale', StandardScaler())])
    prepare = ColumnTransformer([('numbers', numeric_steps, numbers),
                                 ('categories', OneHotEncoder(handle_unknown='ignore'), categories)])
    return Pipeline([('prepare', prepare), ('model', model)])


def test_kaggle_ideas(train, validation):
    """Сравниваем дополнительные признаки и модели только на validation."""
    options = {
        'Logistic + location / road': LogisticRegression(max_iter=500, random_state=42,
                                      class_weight={1: 3, 2: 1, 3: 2, 4: 8}),
        'Random Forest + location / road': RandomForestClassifier(n_estimators=100, max_depth=18,
                                             min_samples_leaf=5, n_jobs=2, random_state=42,
                                             class_weight={1: 3, 2: 1, 3: 2, 4: 8}),
    }
    rows, models, predictions = [], {}, {}
    for name, estimator in options.items():
        print('Обучаем:', name, flush=True)
        model = make_extended_pipeline(estimator)
        seconds, memory = fit_with_measurement(model, train[EXTENDED_FEATURES], train['Severity'])
        pred = model.predict(validation[EXTENDED_FEATURES])
        rows.append({'model': name, 'accuracy': accuracy_score(validation['Severity'], pred),
                     'f1_macro': f1_score(validation['Severity'], pred, labels=LABELS, average='macro', zero_division=0),
                     'train_seconds': seconds, 'peak_process_ram_mib': memory})
        models[name], predictions[name] = model, pred
    return pd.DataFrame(rows), models, predictions
