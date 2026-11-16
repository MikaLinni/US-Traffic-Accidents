"""Аудит и оптимизация готовых Parquet без повторного чтения CSV."""
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

NUMBERS = ['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)',
           'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
           'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
CATEGORIES = ['Source', 'State', 'Country', 'Timezone', 'Weather_Condition',
              'Wind_Direction', 'Sunrise_Sunset', 'Civil_Twilight',
              'Nautical_Twilight', 'Astronomical_Twilight']
BOOLEANS = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
            'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
            'Traffic_Signal', 'Turning_Loop']


def optimize_frame(frame):
    """Сохраняем координаты float64; погода float32 с замером погрешности."""
    result = frame.copy()
    errors = {}
    for name in NUMBERS:
        values = pd.to_numeric(frame[name], errors='coerce').astype('float64')
        # Координаты не округляем, чтобы не сместить редкие близкие точки.
        target = values if name.endswith(('_Lat', '_Lng')) else values.astype('float32')
        diff = (values - target.astype('float64')).abs()
        errors[name] = float(diff.max()) if diff.notna().any() else 0.0
        result[name] = target
    result['Severity'] = pd.to_numeric(frame['Severity'], errors='coerce').astype('Int8')
    for name in CATEGORIES + BOOLEANS:
        # Category сохраняется в Parquet как dictionary, а не обратно как string.
        result[name] = frame[name].astype('category')
    return result, errors


def audit_and_optimize(source, destination, report_path):
    """Проверяем все строки и сохраняем отдельно оптимизированную копию."""
    source, destination, report_path = map(Path, (source, destination, report_path))
    if report_path.exists() and destination.exists():
        report = json.loads(report_path.read_text())
        if report.get('complete'):
            return report
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Незавершённая копия: используйте новую папку назначения')
    files = sorted(source.glob('year=*/part-*.parquet'))
    if not files:
        raise FileNotFoundError(source)
    report = dict(rows=0, files=len(files), missing={}, checks={}, memory_before_bytes=0,
                  numeric64_memory_bytes=0, memory_after_bytes=0, max_abs_rounding_error={})
    started = time.perf_counter()
    def count(name, mask):
        report['checks'][name] = report['checks'].get(name, 0) + int(mask.sum())
    with tempfile.TemporaryDirectory() as tmp:
        with sqlite3.connect(Path(tmp) / 'seen.db') as db:
            db.executescript('CREATE TABLE ids (value TEXT PRIMARY KEY) WITHOUT ROWID;'
                             'CREATE TABLE events (value TEXT PRIMARY KEY) WITHOUT ROWID;')
            for i, file in enumerate(files, 1):
                frame = pd.read_parquet(file)
                report['rows'] += len(frame)
                report['memory_before_bytes'] += int(frame.memory_usage(deep=True).sum())
                for name, n in frame.isna().sum().items():
                    report['missing'][name] = report['missing'].get(name, 0) + int(n)
                for table, series in [('ids', frame.ID.dropna().astype(str)),
                                      ('events', pd.util.hash_pandas_object(frame[['Start_Time','Start_Lat','Start_Lng','End_Time','Severity']], index=False).astype(str))]:
                    before = db.total_changes
                    db.executemany(f'INSERT OR IGNORE INTO {table} VALUES (?)', ((s,) for s in series))
                    key = 'duplicate_id_extra_rows' if table == 'ids' else 'candidate_event_duplicate_extra_rows'
                    report['checks'][key] = report['checks'].get(key, 0) + len(series) - (db.total_changes-before)
                db.commit()
                start = pd.to_datetime(frame.Start_Time, format='mixed', errors='coerce')
                end = pd.to_datetime(frame.End_Time, format='mixed', errors='coerce')
                weather = pd.to_datetime(frame.Weather_Timestamp, format='mixed', errors='coerce')
                count('missing_or_invalid_start', start.isna())
                count('missing_or_invalid_end', end.isna())
                count('end_before_start', end < start)
                count('start_outside_2016_2023', start.notna() & ~start.dt.year.between(2016, 2023))
                count('weather_after_start', weather > start)
                count('weather_older_than_3h', (start-weather) > pd.Timedelta(hours=3))
                count('invalid_or_missing_severity', ~pd.to_numeric(frame.Severity, errors='coerce').isin([1,2,3,4]))
                bounds = {'Start_Lat':(-90,90), 'End_Lat':(-90,90), 'Start_Lng':(-180,180),
                          'End_Lng':(-180,180), 'Humidity(%)':(0,100), 'Temperature(F)':(-100,140),
                          'Wind_Chill(F)':(-150,150), 'Distance(mi)':(0,np.inf),
                          'Visibility(mi)':(0,np.inf), 'Wind_Speed(mph)':(0,np.inf),
                          'Precipitation(in)':(0,np.inf), 'Pressure(in)':(0,np.inf)}
                for name, (low, high) in bounds.items():
                    x = pd.to_numeric(frame[name], errors='coerce')
                    count('invalid_' + name, frame[name].notna() & (x.isna() | ~np.isfinite(x) | ~x.between(low, high)))
                for name in BOOLEANS:
                    count('invalid_' + name, frame[name].notna() & ~frame[name].isin(['True','False']))
                numeric64 = frame.copy()
                for name in NUMBERS + ['Severity']:
                    numeric64[name] = pd.to_numeric(frame[name], errors='coerce').astype('float64')
                report['numeric64_memory_bytes'] += int(numeric64.memory_usage(deep=True).sum())
                optimized, errors = optimize_frame(frame)
                report['memory_after_bytes'] += int(optimized.memory_usage(deep=True).sum())
                for name, error in errors.items():
                    report['max_abs_rounding_error'][name] = max(report['max_abs_rounding_error'].get(name, 0), error)
                target = destination / file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                optimized.to_parquet(target, index=False)
                restored = pd.read_parquet(target)
                if len(restored) != len(frame) or not restored.ID.equals(frame.ID):
                    raise AssertionError('При сохранении изменились строки или ID')
                if i % 20 == 0 or i == len(files):
                    print(f'Аудит и downcast: {i}/{len(files)}, строк {report["rows"]:,}', flush=True)
    report.update(complete=True, elapsed_seconds=time.perf_counter()-started,
                  source_parquet_bytes=sum(f.stat().st_size for f in files),
                  optimized_parquet_bytes=sum(f.stat().st_size for f in destination.glob('year=*/part-*.parquet')),
                  memory_note='Сумма памяти DataFrame по файлам, НЕ пиковая RAM процесса.',
                  duplicate_note='Кандидаты событий: совпадение Start_Time/Start_Lat/Start_Lng/End_Time/Severity, 64-bit hash. Это не доказанные дубли; автоматически не удаляются.')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/processed/full_parquet')
    parser.add_argument('--destination', default='data/processed/optimized_parquet')
    parser.add_argument('--report', default='outputs/optimization_audit.json')
    args = parser.parse_args()
    audit_and_optimize(args.source, args.destination, args.report)
