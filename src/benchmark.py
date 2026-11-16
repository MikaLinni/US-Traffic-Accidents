"""Сравнение Pandas и Polars на подсчёте ДТП по Severity."""
import json
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import pandas as pd
import psutil


# Этот код запускается отдельно для каждого измерения.
# Так использование памяти одной библиотеки не смешивается с другой.
WORKER_CODE = r'''
import json
import sys
from pathlib import Path

engine, folder = sys.argv[1], sys.argv[2]
files = sorted(str(p) for p in Path(folder).glob('year=*/part-*.parquet'))
if not files:
    raise ValueError('Файлы Parquet не найдены')

if engine == 'Pandas':
    import pandas as pd
    counts = {}
    for file in files:
        frame = pd.read_parquet(file, columns=['Severity'])
        for level, count in frame['Severity'].dropna().value_counts().items():
            counts[str(level)] = counts.get(str(level), 0) + int(count)
elif engine == 'Polars':
    import polars as pl
    # Lazy: Polars сначала строит план вычислений, а выполняет его в collect.
    result = (pl.scan_parquet(files, hive_partitioning=False)
                .select('Severity')
                .filter(pl.col('Severity').is_not_null())
                .group_by('Severity').len().collect(engine='streaming'))
    counts = {str(level): int(count) for level, count in result.iter_rows()}
else:
    raise ValueError('Неизвестная библиотека')
print(json.dumps(counts))
'''


def measure_tool(engine, folder):
    """Запускает отдельный процесс и измеряет время и приблизительный максимум RAM."""
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as output, tempfile.TemporaryFile(mode='w+', encoding='utf-8') as errors:
        process = subprocess.Popen(
            [sys.executable, '-c', WORKER_CODE, engine, str(Path(folder).resolve())],
            stdout=output, stderr=errors, text=True,
        )
        tracked = psutil.Process(process.pid)
        peak_bytes = 0
        while process.poll() is None:
            try:
                peak_bytes = max(peak_bytes, tracked.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.02)
        output.seek(0)
        errors.seek(0)
        stdout, stderr = output.read(), errors.read()
    seconds = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f'{engine}: {stderr}')
    return {'tool': engine, 'seconds': seconds, 'peak_memory_mb': peak_bytes / 1024**2,
            'counts': json.loads(stdout)}


def compare_tools(folder):
    """Проводит два измерения для каждого инструмента в разном порядке."""
    measurements = []
    for run, order in enumerate([['Pandas', 'Polars'], ['Polars', 'Pandas']], start=1):
        for engine in order:
            print(f'Запуск {run}: {engine}', flush=True)
            result = measure_tool(engine, folder)
            result['run'] = run
            measurements.append(result)
    first_counts = measurements[0]['counts']
    if not all(item['counts'] == first_counts for item in measurements):
        raise ValueError('Подсчёты не совпадают! Проверьте данные и запросы.')
    table = pd.DataFrame([{key: value for key, value in item.items() if key != 'counts'}
                          for item in measurements])
    return table, first_counts
