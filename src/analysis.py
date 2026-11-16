"""Simple analysis of Parquet files, one file at a time."""
from pathlib import Path

import pandas as pd


def add_counts(total, current):
    """Add counts from one file to the counts from previous files."""
    for label, count in current.items():
        total[label] = total.get(label, 0) + int(count)


def analyze_files(folder):
    files = sorted(Path(folder).glob('year=*/part-*.parquet'))
    if not files:
        raise FileNotFoundError(f'No Parquet files found in {folder}')

    counts = {'severity': {}, 'state': {}, 'hour': {}, 'weather': {}, 'year': {}}
    columns = ['Severity', 'State', 'Start_Time', 'Weather_Condition']
    for file in files:
        data = pd.read_parquet(file, columns=columns)
        add_counts(counts['severity'], data['Severity'].dropna().value_counts())
        add_counts(counts['state'], data['State'].dropna().value_counts())
        add_counts(counts['weather'], data['Weather_Condition'].dropna().value_counts())
        # The hours and years are in fixed positions in the original date string.
        add_counts(counts['hour'], data['Start_Time'].str.slice(11, 13).dropna().value_counts())
        add_counts(counts['year'], data['Start_Time'].str.slice(0, 4).dropna().value_counts())

    results = {}
    for name, values in counts.items():
        table = pd.DataFrame(list(values.items()), columns=[name, 'accidents'])
        results[name] = table.sort_values('accidents', ascending=False).reset_index(drop=True)
    return results
