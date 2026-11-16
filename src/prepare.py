"""Chunked conversion of the US Accidents ZIP to year-partitioned Parquet."""
import json
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@contextmanager
def open_dataset(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith('.csv')]
            if len(members) != 1:
                raise ValueError('The archive must contain exactly one CSV file')
            with archive.open(members[0]) as source:
                yield source
    else:
        with path.open('rb') as source:
            yield source


def prepare_data(archive_path, output_dir, chunk_size=100_000, max_rows=None):
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    if chunk_size < 1 or (max_rows is not None and max_rows < 1):
        raise ValueError('chunk_size and max_rows must be positive')
    # Refuse to mix a previous run with the current dataset.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {'rows': 0, 'missing': {}, 'severity': {}, 'invalid_start_time': 0,
              'invalid_severity': 0, 'memory_before_bytes': 0, 'memory_after_bytes': 0}
    started = time.perf_counter()
    with open_dataset(archive_path) as source:
        reader = pd.read_csv(source, chunksize=chunk_size, dtype=str, keep_default_na=True,
                             nrows=max_rows, low_memory=False)
        for number, chunk in enumerate(reader):
            totals['rows'] += len(chunk)
            totals['memory_before_bytes'] += int(chunk.memory_usage(deep=True).sum())
            for col, value in chunk.isna().sum().items():
                totals['missing'][col] = totals['missing'].get(col, 0) + int(value)
            for value, count in chunk['Severity'].fillna('<missing>').value_counts().items():
                totals['severity'][value] = totals['severity'].get(value, 0) + int(count)
            dates = pd.to_datetime(chunk['Start_Time'], errors='coerce', format='mixed')
            # Older Pandas versions do not support format='mixed'.
            if dates.isna().all() and chunk['Start_Time'].notna().any():
                dates = pd.to_datetime(chunk['Start_Time'], errors='coerce')
            totals['invalid_start_time'] += int((dates.isna() & chunk['Start_Time'].notna()).sum())
            totals['invalid_severity'] += int((~chunk['Severity'].isin(['1', '2', '3', '4']) & chunk['Severity'].notna()).sum())
            for col in ['Severity', 'State', 'Sunrise_Sunset', 'Weather_Condition']:
                chunk[col] = chunk[col].astype('category')
            totals['memory_after_bytes'] += int(chunk.memory_usage(deep=True).sum())
            years = dates.dt.year.fillna(0).astype(int)
            for year in sorted(years.unique()):
                target = output_dir / f'year={year}'
                target.mkdir(exist_ok=True)
                part = chunk.loc[years == year].copy()
                # Strings ensure a stable schema across chunks and partitions.
                for col in part.columns:
                    part[col] = part[col].astype('string')
                table = pa.Table.from_pandas(part, preserve_index=False)
                pq.write_table(table, target / f'part-{number:05d}.parquet', compression='snappy')
            print(f'Processed {totals["rows"]:,} rows', flush=True)
    totals['elapsed_seconds'] = round(time.perf_counter() - started, 2)
    totals['scope'] = 'full archive' if max_rows is None else f'first {max_rows} rows at most'
    totals['memory_note'] = 'Sum of per-chunk dataframe memory, not peak process memory.'
    (output_dir / 'audit.json').write_text(json.dumps(totals, indent=2, ensure_ascii=False), encoding='utf-8')
    return totals
