"""Упаковывает текущие модули в один переносимый ноутбук Colab."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted((root/'src').glob('*.py')):
        bundle.write(path,'src/'+path.name)
    bundle.writestr('src/__init__.py','"""Модули проекта US Accidents."""\n')
    bundle.write(root/'requirements.txt','requirements.txt')
    bundle.write(root/'docs/FINAL_PROTOCOL.md','docs/FINAL_PROTOCOL.md')
archive_path = root/'deliverables/project_code.zip'
archive_path.parent.mkdir(parents=True, exist_ok=True)
archive_path.write_bytes(buffer.getvalue())
digest=hashlib.sha256(buffer.getvalue()).hexdigest()
c=[]
def md(s): c.append(nbf.v4.new_markdown_cell(s))
def code(s): c.append(nbf.v4.new_code_cell(s))
md('''# US Traffic Accidents Patterns and Severity Prediction
Финальный проект №42, Programming for Data Analysis and Database.

Запускайте сверху вниз. Для Colab нужны этот ноутбук и небольшой `project_code.zip`: выберите ZIP один раз в ячейке загрузки модулей. Датасет загружается автоматически. Код остаётся в обычных Python-файлах `src`, без Base64. Полный набор содержит 7 728 394 записи. Severity описывает влияние ДТП на движение.

Обработка всех данных и обучение требуют времени, интернета и свободного диска. Локально используется уже подготовленный Parquet. Все результаты сохраняются в папке проекта. Предыдущие исследования сохранены отдельно в репозитории, а здесь выполняется только итоговый протокол.''')
code('''from pathlib import Path
import os, sys, json, subprocess, zipfile
IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        pass
# Для постоянного кэша можно включить Drive; это необязательно для запуска.
USE_DRIVE = False
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
ROOT = (Path('/content/drive/MyDrive/US_Traffic_Project') if USE_DRIVE else Path('/content/traffic_project')) if IN_COLAB else Path.cwd()
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mpl_cache')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
''')
code('''# Один раз выберите project_code.zip. Датасет вручную загружать не нужно.
archive = ROOT / 'project_code.zip'
if not (ROOT / 'src/pipeline.py').exists():
    if not archive.exists():
        from google.colab import files
        uploaded = files.upload()
        if 'project_code.zip' not in uploaded:
            raise FileNotFoundError('Выберите файл project_code.zip')
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (ROOT / name).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Недопустимый путь в ZIP')
        bundle.extractall(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
''')
code('''# В Colab установка выполняется автоматически. Локально используйте requirements.txt.
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
import pandas as pd
from IPython.display import display, Image, HTML
from src.prepare import prepare_data
from src.download import download_dataset
from src.pipeline import run_analysis
from src.optimize import audit_and_optimize
from src.final_modeling import run
from src.menu import run_menu
PROCESSED = ROOT / 'data/processed'
PARQUET = PROCESSED / 'full_parquet'
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
DATABASE = PROCESSED / 'accidents.db'
''')
md('''## Подготовленные данные и повторное использование
Если Parquet уже есть, CSV повторно не скачивается и не обрабатывается. При первом запуске в Colab загружается публичная копия CSV с закреплённой версией и SHA-256. При ошибке скачивания программа останавливается и сообщает ошибку. Данные в `/content` временные. Для сохранения между сеансами включите `USE_DRIVE` в настройке выше.''')
code('''audit_path = PARQUET / 'audit.json'
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    if audit['rows'] != 7728394:
        raise ValueError('Ожидается полный набор 7 728 394 записи')
else:
    csv = download_dataset(ROOT / 'data/raw/US_Accidents_March23.csv')
    audit = prepare_data(csv, PARQUET)
import pyarrow.parquet as pq
assert sum(pq.ParquetFile(p).metadata.num_rows for p in PARQUET.glob('year=*/part-*.parquet')) == audit['rows']
print('Строк:', audit['rows'])
''')
md('''## Аудит качества и числовой downcast
Читаем готовые Parquet по одному файлу. Проверяем пропуски, дубликаты ID и кандидатов событий, даты и порядок дат, координаты и погодные числа. Все исходные строки сохраняем. Не удаляем экстремальную погоду только по IQR. Невозможные ML-признаки превращаются в пропуски, медианы учатся на train.

Отдельная оптимизированная копия хранит Severity как Int8, погоду float32, координаты float64, повторяющиеся строки category. Измеряем память до/после и погрешность чисел. Сумма памяти чанков не равна пиковой RAM.''')
code('''optimization = audit_and_optimize(PARQUET, PROCESSED / 'optimized_parquet', OUT / 'optimization_audit.json')
display(pd.Series(optimization['checks']).to_frame('Количество'))
display(pd.Series({k:optimization[k] for k in ['memory_before_bytes','numeric64_memory_bytes','memory_after_bytes','source_parquet_bytes','optimized_parquet_bytes']}).to_frame('Байт'))
print('Максимальные погрешности:', optimization['max_abs_rounding_error'])
''')
md('''## SQLite и агрегаты
Схема: `states(state_id PK, state_code UNIQUE)` связана 1:N с `accidents(accident_id PK, state_id FK, severity, start_time, weather)`. Таблица `project_info` хранит число загруженных строк. Код `src/database.py` содержит восемь SQL-запросов: JOIN, GROUP BY, HAVING, CTE, подзапрос и RANK OVER. Результаты загружаем в Pandas и сохраняем в CSV. Агрегаты сохраняем в SQLite и Parquet.''')
md(''.join(json.loads((root/'docs/experiments/US_Accidents_Exploration.ipynb').read_text())['cells'][21]['source']))
md(''.join(json.loads((root/'docs/experiments/US_Accidents_Exploration.ipynb').read_text())['cells'][24]['source']))
code('''results = run_analysis(PARQUET, DATABASE, OUT)
from src.database import QUERIES
for name, query in QUERIES.items():
    print(name, query.strip())
    display(pd.read_csv(OUT / 'sql' / (name + '.csv')).head(10))
''')
md('''## EDA и интерпретация
Графики показывают зарегистрированные ДТП. Без данных о пробеге по штатам и погоде нельзя оценить риск ДТП или причинный эффект. Пики по часам могут отражать интенсивность движения и покрытие источников. 2023 год неполный. Дисбаланс Severity делает Accuracy недостаточной метрикой.''')
code('''import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

def show_plot():
    # Явно показываем PNG, даже если модуль сохранения графиков включил backend Agg.
    import io
    image_buffer = io.BytesIO()
    plt.gcf().savefig(image_buffer, format='png', dpi=140, bbox_inches='tight')
    display(Image(data=image_buffer.getvalue()))
    plt.close()

display(pd.read_csv(OUT / 'numeric_description.csv'))
print((OUT / 'figures/interpretation.txt').read_text())
''')
# Возвращаем отдельные интерактивные графики прежнего ноутбука.
old = json.loads((root/'docs/experiments/US_Accidents_Exploration.ipynb').read_text())
graph_notes = [
    ('Распределение Severity', 'Показывает дисбаланс метки: преобладает Severity 2. Это влияние на движение, а не тяжесть травм.'),
    ('Десять штатов', 'Сравниваем количество зарегистрированных событий. Без пробега и интенсивности движения это не сравнение риска.'),
    ('Местный час начала', 'Пики могут отражать часы поездок и покрытие источников. График не доказывает опасность конкретного часа.'),
    ('Количество записей по годам', 'Покрытие данных меняется со временем. 2023 год неполный, его нельзя напрямую сравнивать с полными годами.'),
    ('Частые погодные условия', 'Пропуски погоды исключены. Частота ДТП при погоде не равна риску ДТП: нет числа поездок в этих условиях.'),
]
for index, (title, interpretation) in zip(range(15, 20), graph_notes):
    md('### ' + title + '\n\n' + interpretation)
    code(''.join(old['cells'][index]['source']))
md('### Пропуски в исходных данных\n\nПропуск не означает ноль. Показываем десять столбцов с наибольшим числом пропусков до очистки.')
code(''.join(old['cells'][32]['source']).replace('plt.show()', 'show_plot()'))
md('''## Две гипотезы и размер связи
H0: погода и Severity независимы. H0: день/ночь и Severity независимы. Используем chi-square и Cramér V. При миллионах записей p-value может округляться до 0 из-за машинной точности. Это не вероятность, равная математическому нулю. Малый V указывает на слабую связь. Зависимость соседних событий ограничивает статистическую интерпретацию.''')
code('''quality_file = OUT / 'quality_hypotheses.json'
if not quality_file.exists():
    subprocess.run([sys.executable, '-m', 'src.pipeline', 'quality'], check=True)
hypotheses = json.loads(quality_file.read_text())
display(pd.DataFrame({k:v for k,v in hypotheses.items() if k != 'quality'}).T)
''')
md('### Таблицы проверки гипотез\n\nВ таблицах показано число наблюдений каждого класса. По ним рассчитаны chi-square и Cramér V.')
code('''weather_table = pd.read_csv(OUT / 'weather_severity.csv', index_col=0)
day_table = pd.read_csv(OUT / 'day_severity.csv', index_col=0)
display(weather_table)
display(day_table)
''')
md(''.join(old['cells'][35]['source']))
code(''.join(old['cells'][36]['source']).replace('plt.show()', 'show_plot()'))
md('Сравниваем доли внутри дня и ночи. Различия описывают состав зарегистрированных ДТП, но не доказывают причинный эффект освещённости.')
md('''## Pandas против Polars
Одна задача: количество ДТП по Severity на одинаковых файлах. Pandas считает инкрементально, Polars использует lazy streaming. По два запуска в разном порядке, отдельные процессы, приблизительная пиковая RSS. Кэш ОС и оборудование влияют на время.''')
code('''from src.benchmark import compare_tools
benchmark_file = OUT / 'pandas_polars_benchmark.csv'
if not benchmark_file.exists():
    table, counts = compare_tools(PARQUET)
    table.to_csv(benchmark_file,index=False)
display(pd.read_csv(benchmark_file))
''')
md('''## Финальный протокол моделей
Train: до 2021 года, старые 30% по хешу ID. Validation: 2021, те же 30%. Test: 2022–2023, другие 10% по хешу [3000,4000), вне прежних модельных экспериментов. Полный набор уже использован в EDA, поэтому абсолютную независимость не заявляем. Подробности в `docs/FINAL_PROTOCOL.md`.

Baseline Dummy обучается на естественных долях. Logistic и HGB используют одинаковые уникальные train-строки, до 50 000 каждого класса без повторов. Варианты весов HGB сравниваются только по validation natural F1-macro. После фиксации победителя test оценивается один раз. Сохранённые результаты используются повторно.

`stratify` сохраняет доли при случайном разбиении. Веса меняют вклад классов при обучении. Равная оценочная выборка меняет частоты на проверке. Это три разные операции. Временной split не использует stratify.''')
code('''scores = run(PARQUET, OUT / 'final')
display(pd.read_csv(OUT / 'final/validation_scores.csv'))
print('Выбор зафиксирован до test:', (OUT / 'final/selection_before_test.json').read_text())
display(scores)
''')
md('''## Метрики слабых классов и ошибки
В таблицах каждого класса support означает число настоящих наблюдений. Сравнивайте precision и recall: увеличение recall редкого класса может сопровождаться большим количеством ложных тревог. Если Dummy выигрывает на test, это сохраняем как результат, не меняя модель после просмотра test.''')
code('''selection = json.loads((OUT / 'final/selection_before_test.json').read_text())
selected = selection['selected_hgb']
display(pd.read_csv(OUT / f'final/{selected}_test_classes.csv',index_col=0))
display(pd.read_csv(OUT / f'final/{selected}_test_confusion.csv',index_col=0))
display(pd.read_csv(OUT / f'final/{selected}_test_weather_errors.csv'))
''')
md('''## Original Contribution / Additional Experiments
Одна HGB с погодой и без неё использует те же строки, настройки и веса. Сравниваем F1-macro, Accuracy и время. Дополнительно анализируем ошибки по погодным группам с n и support всех классов. Разница метрик не доказывает причинное влияние погоды.

В локальном проверенном запуске временной test оказался сложным: HGB F1-macro около 0,208, Dummy около 0,239. Прежние 0,615 на равной validation и 0,315 на естественной validation относятся к другому случайному эксперименту. Их нельзя выдавать за test.''')
code('''display(scores[scores.model.str.startswith('hgb')])
validation_scores = pd.read_csv(OUT / 'final/validation_scores.csv')
display(validation_scores[validation_scores.model.isin([selected, selected + '_no_weather'])])
''')
md('''## Интерактивная панель анализа
Выберите год, штат и Severity и нажмите «Показать результаты». Панель показывает количество записей, доли классов и интерактивные графики. «Сохранить CSV» сохраняет агрегаты показанной выборки. Используется готовая SQLite; обучение моделей для панели не требуется. Консольное меню src/menu.py остаётся доступным для запуска из терминала.''')
code('''from src.dashboard import show_dashboard
show_dashboard(DATABASE, OUT / 'dashboard')
''')
md('''## Сохранение результатов
Создаём ZIP с таблицами, графиками, моделью и метриками. Сырые и обработанные данные в ZIP не входят. В Colab результат можно скачать одной кнопкой. Для постоянного повторного использования Parquet применяйте Google Drive из настройки в начале.''')
code('''import shutil
archive = shutil.make_archive(str(ROOT / 'traffic_results'), 'zip', OUT)
print('Результаты:', archive)
if IN_COLAB:
    from google.colab import files
    files.download(archive)
''')
notebook=nbf.v4.new_notebook(cells=c,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13'}})
nbf.write(notebook,root/'US_Accidents_Colab.ipynb')
(root/'docs/COLAB_MODULES_CELL.py').write_text(c[2].source + '\n')
print('Ячеек',len(c),'Архив модулей',archive_path,'SHA',digest)
