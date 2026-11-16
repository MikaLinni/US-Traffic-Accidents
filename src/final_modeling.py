"""Фиксированное сравнение моделей и однократная финальная оценка."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from src.modeling import LABELS, WEATHER_NUMBERS, fit_with_measurement
from src.quality_hypotheses import weather_group
from src.stratified_experiment import balanced_evaluation

NUMERIC = ['hour', 'day_of_week', 'month', 'Start_Lat', 'Start_Lng']
CATEGORICAL = ['State', 'Sunrise_Sunset', 'Crossing', 'Junction', 'Traffic_Signal', 'Stop']
WEATHER = WEATHER_NUMBERS + ['weather_group']


def features(with_weather=True):
    return NUMERIC + CATEGORICAL + (WEATHER if with_weather else [])


def clean_features(frame):
    """Удаляем невозможные значения признаков, не удаляя редкие классы."""
    frame = frame.copy()
    frame['start'] = pd.to_datetime(frame.Start_Time, format='mixed', errors='coerce')
    frame['Severity'] = pd.to_numeric(frame.Severity, errors='coerce')
    valid = frame.ID.notna() & frame.start.notna() & frame.Severity.isin(LABELS)
    frame = frame.loc[valid].copy()
    frame['Severity'] = frame.Severity.astype('int8')
    frame['hour'] = frame.start.dt.hour.astype('int8')
    frame['day_of_week'] = frame.start.dt.dayofweek.astype('int8')
    frame['month'] = frame.start.dt.month.astype('int8')
    weather_time = pd.to_datetime(frame.Weather_Timestamp, format='mixed', errors='coerce')
    timely = (frame.start-weather_time).between(pd.Timedelta(0), pd.Timedelta(hours=3))
    limits = {'Start_Lat':(-90,90), 'Start_Lng':(-180,180), 'Temperature(F)':(-100,140),
              'Humidity(%)':(0,100), 'Visibility(mi)':(0,np.inf)}
    for name, (low, high) in limits.items():
        x = pd.to_numeric(frame[name], errors='coerce').astype(float)
        frame[name] = x.where(np.isfinite(x) & x.between(low, high))
        if name in WEATHER_NUMBERS:
            frame[name] = frame[name].where(timely)
    frame['weather_group'] = frame.Weather_Condition.astype('object').map(weather_group).where(timely, 'Unknown')
    for name in CATEGORICAL:
        x = frame[name].astype('string')
        if name in ['Crossing','Junction','Traffic_Signal','Stop']:
            x = x.where(x.isin(['True','False']))
        frame[name] = x.fillna('Unknown').astype(str)
    return frame


def prepare_splits(folder, output):
    """Новые test-ID берём вне всех прежних модельных выборок."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    columns = ['ID','Severity','Start_Time','Weather_Timestamp','Weather_Condition'] + ['Start_Lat','Start_Lng'] + CATEGORICAL + WEATHER_NUMBERS
    pieces = {'train':[], 'validation':[], 'test':[]}
    input_files = sorted(Path(folder).glob('year=*/part-*.parquet'))
    if not input_files:
        raise FileNotFoundError(folder)
    for file in input_files:
        data = pd.read_parquet(file, columns=columns)
        h = pd.util.hash_pandas_object(data.ID.astype('string'), index=False) % 10000
        year = pd.to_numeric(data.Start_Time.str[:4], errors='coerce')
        masks = {'train':(year < 2021) & (h < 3000),
                 'validation':(year == 2021) & (h < 3000),
                 'test':(year >= 2022) & h.between(3000,3999)}
        for name, mask in masks.items():
            if mask.any():
                pieces[name].append(clean_features(data.loc[mask]))
    frames = {k:pd.concat(v, ignore_index=True) for k,v in pieces.items()}
    ids = {k:set(v.ID) for k,v in frames.items()}
    for name, frame in frames.items():
        if frame.ID.duplicated().any():
            raise ValueError(f'Повтор ID в {name}')
        if not set(LABELS).issubset(set(frame.Severity)):
            raise ValueError(f'Отсутствует класс в {name}')
    if any(ids[a] & ids[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Пересечение частей по ID')
    # Сохраняем естественный train отдельно для baseline.
    frames['train'].to_parquet(output / 'train_natural.parquet', index=False)
    capped = []
    for level in LABELS:
        group = frames['train'].loc[frames['train'].Severity == level]
        capped.append(group.sample(n=min(50000,len(group)), random_state=42))
    frames['train'] = pd.concat(capped).sample(frac=1,random_state=42).reset_index(drop=True)
    summary = {}
    for name, frame in frames.items():
        frame.to_parquet(output / f'{name}.parquet', index=False)
        summary[name] = {'rows':len(frame), 'unique_ids':int(frame.ID.nunique()),
                         'class_counts':frame.Severity.value_counts().sort_index().to_dict()}
    (output / 'split_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def pipeline(kind, with_weather=True):
    numbers = NUMERIC + (WEATHER_NUMBERS if with_weather else [])
    categories = CATEGORICAL + (['weather_group'] if with_weather else [])
    if kind == 'dummy':
        return DummyClassifier(strategy='most_frequent')
    hgb = kind.startswith('hgb')
    prep = ColumnTransformer([
        ('numbers', Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),
                              ('scale',StandardScaler())]), numbers),
        ('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=not hgb),categories)])
    model = (HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=31,learning_rate=0.1,
                                           early_stopping=False,random_state=42) if hgb else
             LogisticRegression(max_iter=700,random_state=42))
    return Pipeline([('prepare',prep),('model',model)])


def worker(data_dir, output, kind, with_weather=True):
    """Каждая модель обучается в отдельном процессе для сравнимой RAM."""
    output, data_dir = Path(output), Path(data_dir)
    train = pd.read_parquet(data_dir / ('train_natural.parquet' if kind == 'dummy' else 'train.parquet'))
    validation = pd.read_parquet(data_dir / 'validation.parquet')
    model = pipeline(kind,with_weather)
    weight = compute_sample_weight('balanced', train.Severity)
    if kind == 'hgb_sqrt':
        weight = np.sqrt(weight)
    if kind == 'hgb_none':
        weight = np.ones(len(train))
    # fit_with_measurement принимает model.fit без kwargs: обёртка передаёт веса.
    class WeightedFit:
        def fit(self,x,y):
            if kind == 'dummy':
                return model.fit(x,y)
            return model.fit(x,y,model__sample_weight=weight)
    import psutil
    before = psutil.Process().memory_info().rss / 1024**2
    with threadpool_limits(limits=4):
        seconds, peak = fit_with_measurement(WeightedFit(), train[features(with_weather)],train.Severity)
        started = time.perf_counter()
        pred = model.predict(validation[features(with_weather)])
        prediction_seconds = time.perf_counter()-started
    name = kind if with_weather else kind+'_no_weather'
    joblib.dump(model, output / f'{name}.joblib')
    pd.DataFrame({'ID':validation.ID,'prediction':pred}).to_parquet(output / f'{name}_validation_predictions.parquet',index=False)
    result = dict(model=name,accuracy=accuracy_score(validation.Severity,pred),
                  f1_macro=f1_score(validation.Severity,pred,labels=LABELS,average='macro',zero_division=0),
                  train_seconds=seconds, validation_predict_seconds=prediction_seconds,
                  peak_process_ram_mib=peak, before_fit_ram_mib=before,
                  train_rows=len(train), train_unique_ids=int(train.ID.nunique()))
    (output / f'{name}_validation.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


def evaluate(model, frame, output, name):
    """Сохраняем все классы, матрицу ошибок и ошибки по погоде."""
    output = Path(output)
    with_weather = not name.endswith('_no_weather')
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        pred = model.predict(frame[features(with_weather)])
    seconds = time.perf_counter()-started
    pd.DataFrame(classification_report(frame.Severity,pred,labels=LABELS,output_dict=True,zero_division=0)).T.to_csv(output/f'{name}_test_classes.csv')
    pd.DataFrame(confusion_matrix(frame.Severity,pred,labels=LABELS),index=LABELS,columns=LABELS).to_csv(output/f'{name}_test_confusion.csv')
    scored = frame[['ID','Severity','weather_group']].copy()
    scored['prediction'] = pred
    rows = []
    for group, part in scored.groupby('weather_group'):
        row = dict(weather_group=group,n=len(part),accuracy=accuracy_score(part.Severity,part.prediction),
                   f1_macro=f1_score(part.Severity,part.prediction,labels=LABELS,average='macro',zero_division=0))
        for level in LABELS:
            row[f'support_{level}'] = int((part.Severity==level).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/f'{name}_test_weather_errors.csv',index=False)
    # Равную оценку получаем подвыборкой уже предсказанных строк, без повторов.
    equal = balanced_evaluation(scored)
    return dict(model=name,accuracy_natural=accuracy_score(scored.Severity,pred),
                f1_macro_natural=f1_score(scored.Severity,pred,labels=LABELS,average='macro',zero_division=0),
                accuracy_equal_classes=accuracy_score(equal.Severity,equal.prediction),
                f1_macro_equal_classes=f1_score(equal.Severity,equal.prediction,labels=LABELS,average='macro',zero_division=0),
                test_rows=len(frame), equal_test_rows=len(equal),predict_seconds=seconds)


def run(folder, output='outputs/final'):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'test_scores.csv').exists():
        print('Финальная оценка уже выполнена. Используем сохранённые результаты.')
        return pd.read_csv(output/'test_scores.csv')
    data_dir = output/'splits'
    if not (data_dir/'split_summary.json').exists():
        print(prepare_splits(folder,data_dir),flush=True)
    kinds = ['dummy','logistic','hgb_balanced','hgb_sqrt','hgb_none']
    for kind in kinds:
        if not (output/f'{kind}_validation.json').exists():
            subprocess.run([sys.executable,'-m','src.final_modeling','--worker',kind,'--output',str(output)],check=True)
    scores = pd.DataFrame([json.loads((output/f'{kind}_validation.json').read_text()) for kind in kinds])
    hgb = scores[scores.model.str.startswith('hgb')].sort_values('f1_macro',ascending=False).iloc[0].model
    winner = scores[scores.model != 'dummy'].sort_values('f1_macro',ascending=False).iloc[0].model
    # Этот файл записывается ДО чтения test и больше не меняет выбор по test.
    freeze = dict(selected_model=winner,selected_hgb=hgb,selection_metric='validation natural F1-macro',
                  protocol='docs/FINAL_PROTOCOL.md',test_used_for_selection=False)
    (output/'selection_before_test.json').write_text(json.dumps(freeze,indent=2))
    if not (output/f'{hgb}_no_weather_validation.json').exists():
        subprocess.run([sys.executable,'-m','src.final_modeling','--worker',hgb,'--no-weather','--output',str(output)],check=True)
    scores = pd.concat([scores,pd.DataFrame([json.loads((output/f'{hgb}_no_weather_validation.json').read_text())])],ignore_index=True)
    scores.to_csv(output/'validation_scores.csv',index=False)
    test = pd.read_parquet(data_dir/'test.parquet')
    results = []
    for name in ['dummy','logistic',hgb,hgb+'_no_weather']:
        model = joblib.load(output/f'{name}.joblib')
        results.append(evaluate(model,test,output,name))
    final = pd.DataFrame(results)
    final.to_csv(output/'test_scores.csv',index=False)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',default='data/processed/full_parquet')
    parser.add_argument('--output',default='outputs/final')
    parser.add_argument('--worker')
    parser.add_argument('--no-weather',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.output)/'splits',args.output,args.worker,not args.no_weather)
    else:
        print(run(args.folder,args.output).to_string(index=False))
