"""Проверка качества и две гипотезы на данных Parquet."""
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def weather_group(value):
    """Упрощённые группы описаний погоды; порядок условий важен."""
    if pd.isna(value):
        return 'Unknown'
    text = str(value).lower()
    if any(word in text for word in ['snow', 'sleet', 'ice', 'freezing', 'hail']):
        return 'Snow / ice'
    if any(word in text for word in ['rain', 'drizzle', 'thunder', 'shower']):
        return 'Rain / storm'
    if any(word in text for word in ['clear', 'fair', 'sunny']):
        return 'Clear / fair'
    if any(word in text for word in ['cloud', 'overcast']):
        return 'Cloudy'
    return 'Other'


def analyze_quality(folder):
    """Читает файлы по очереди и возвращает показатели качества и таблицы гипотез."""
    files = sorted(Path(folder).glob('year=*/part-*.parquet'))
    if not files:
        raise FileNotFoundError('Файлы Parquet не найдены')
    columns = ['ID', 'Severity', 'Start_Time', 'Temperature(F)', 'Weather_Condition', 'Sunrise_Sunset']
    quality = {'rows': 0, 'missing_id': 0, 'duplicate_id_rows': 0,
               'invalid_severity': 0, 'invalid_date': 0,
               'invalid_temperature': 0, 'missing_temperature': 0}
    temperatures = []
    weather_table = pd.DataFrame(0, index=['Clear / fair', 'Cloudy', 'Rain / storm', 'Snow / ice', 'Other'], columns=[1, 2, 3, 4], dtype='int64')
    day_table = pd.DataFrame(0, index=['Day', 'Night'], columns=[1, 2, 3, 4], dtype='int64')

    # Временная SQLite-таблица хранит уже встречавшиеся ID на диске.
    # Так мы считаем повторы между разными файлами, а не только внутри одного чанка.
    with tempfile.TemporaryDirectory() as temp_dir:
        with sqlite3.connect(Path(temp_dir) / 'ids.db') as db:
            db.execute('CREATE TABLE seen_ids (id TEXT PRIMARY KEY) WITHOUT ROWID')
            for index, file in enumerate(files, 1):
                frame = pd.read_parquet(file, columns=columns)
                quality['rows'] += len(frame)
                ids = frame['ID'].dropna().astype(str)
                quality['missing_id'] += int(frame['ID'].isna().sum())
                before = db.total_changes
                db.executemany('INSERT OR IGNORE INTO seen_ids (id) VALUES (?)', ((value,) for value in ids))
                added = db.total_changes - before
                quality['duplicate_id_rows'] += len(ids) - added
                db.commit()

                severity = pd.to_numeric(frame['Severity'], errors='coerce')
                valid_severity = severity.isin([1, 2, 3, 4])
                quality['invalid_severity'] += int((frame['Severity'].notna() & ~valid_severity).sum())
                date = pd.to_datetime(frame['Start_Time'], format='mixed', errors='coerce')
                quality['invalid_date'] += int((frame['Start_Time'].notna() & date.isna()).sum())
                temperature = pd.to_numeric(frame['Temperature(F)'], errors='coerce')
                quality['missing_temperature'] += int(frame['Temperature(F)'].isna().sum())
                # Границы являются правилом проверки, а не доказательством ошибки.
                invalid_temp = frame['Temperature(F)'].notna() & (temperature.isna() | ~temperature.between(-100, 140))
                quality['invalid_temperature'] += int(invalid_temp.sum())
                temperatures.append(temperature[temperature.between(-100, 140)].to_numpy(dtype='float32'))

                groups = frame['Weather_Condition'].map(weather_group)
                t1 = pd.crosstab(groups[valid_severity], severity[valid_severity]).reindex(index=weather_table.index, columns=weather_table.columns, fill_value=0)
                weather_table += t1
                days = frame['Sunrise_Sunset']
                t2 = pd.crosstab(days[valid_severity], severity[valid_severity]).reindex(index=day_table.index, columns=day_table.columns, fill_value=0)
                day_table += t2
                if index % 20 == 0 or index == len(files):
                    print(f'Проверено файлов: {index} из {len(files)}')

    values = np.concatenate(temperatures)
    if len(values):
        q1, q3 = np.percentile(values, [25, 75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        quality['temperature_q1'] = float(q1)
        quality['temperature_q3'] = float(q3)
        quality['temperature_iqr_lower'] = float(lower)
        quality['temperature_iqr_upper'] = float(upper)
        quality['temperature_iqr_outliers'] = int(((values < lower) | (values > upper)).sum())
    return quality, weather_table, day_table


def test_hypothesis(table):
    """Критерий chi-square и размер эффекта Cramer's V."""
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if min(table.shape) < 2:
        raise ValueError('Для проверки нужны как минимум две группы и два уровня Severity')
    chi2, p_value, degrees, expected = chi2_contingency(table)
    sample_size = table.to_numpy().sum()
    effect = np.sqrt(chi2 / (sample_size * min(table.shape[0] - 1, table.shape[1] - 1)))
    return {'observations': int(sample_size), 'chi2': float(chi2), 'p_value': float(p_value),
            'degrees_of_freedom': int(degrees), 'cramers_v': float(effect),
            'expected_below_5': int((expected < 5).sum())}
