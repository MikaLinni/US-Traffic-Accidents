"""Упаковывает текущие модули в один переносимый ноутбук Colab."""
import base64
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
payload=base64.b64encode(buffer.getvalue()).decode()
digest=hashlib.sha256(buffer.getvalue()).hexdigest()
c=[]
def md(s): c.append(nbf.v4.new_markdown_cell(s))
def code(s): c.append(nbf.v4.new_code_cell(s))
md('''# US Traffic Accidents Patterns and Severity Prediction
Финальный проект №42, Programming for Data Analysis and Database.

Запускайте сверху вниз. Одного этого ноутбука достаточно: он разворачивает настоящие Python-модули `src`, устанавливает библиотеки и автоматически получает проверенную копию данных. Полный набор содержит 7 728 394 записи. Severity описывает влияние ДТП на движение.

Обработка всех данных и обучение требуют времени, интернета и свободного диска. Локально используется уже подготовленный Parquet. Все результаты сохраняются в папке проекта. Предыдущие исследования сохранены отдельно в репозитории, а здесь выполняется только итоговый протокол.''')
code('''from pathlib import Path
import os, sys, json, subprocess, base64, hashlib, io, zipfile
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
code(f'''# Это архив исходников проекта, а не скрытая реализация: после распаковки код доступен в src/.
MODULES_B64 = {payload!r}
raw = base64.b64decode(MODULES_B64)
assert hashlib.sha256(raw).hexdigest() == {digest!r}
# В локальном репозитории не заменяем исходники встроенной копией.
if IN_COLAB or not (ROOT / 'src/pipeline.py').exists():
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        for name in bundle.namelist():
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError('Небезопасный путь в архиве модулей')
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
code('''results = run_analysis(PARQUET, DATABASE, OUT)
from src.database import QUERIES
for name, query in QUERIES.items():
    print(name, query.strip())
    display(pd.read_csv(OUT / 'sql' / (name + '.csv')).head(10))
''')
md('''## EDA и интерпретация
Графики показывают зарегистрированные ДТП. Без данных о пробеге по штатам и погоде нельзя оценить риск ДТП или причинный эффект. Пики по часам могут отражать интенсивность движения и покрытие источников. 2023 год неполный. Дисбаланс Severity делает Accuracy недостаточной метрикой.''')
code('''for name in ['severity','hour','year','state','missing']:
    display(Image(filename=str(OUT / 'figures' / (name + '.png'))))
print((OUT / 'figures/interpretation.txt').read_text())
display(pd.read_csv(OUT / 'numeric_description.csv'))
# Интерактивный график Plotly: наведение, масштабирование, сохранение.
import plotly.express as px
px.bar(results['severity'], x='severity', y='accidents', title='Число ДТП по Severity').show()
''')
md('''## Две гипотезы и размер связи
H0: погода и Severity независимы. H0: день/ночь и Severity независимы. Используем chi-square и Cramér V. При миллионах записей p-value может округляться до 0 из-за машинной точности. Это не вероятность, равная математическому нулю. Малый V указывает на слабую связь. Зависимость соседних событий ограничивает статистическую интерпретацию.''')
code('''quality_file = OUT / 'quality_hypotheses.json'
if not quality_file.exists():
    subprocess.run([sys.executable, '-m', 'src.pipeline', 'quality'], check=True)
hypotheses = json.loads(quality_file.read_text())
display(pd.DataFrame({k:v for k,v in hypotheses.items() if k != 'quality'}).T)
''')
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
md('''## Интерактивное меню
Меню работает с SQLite и параметризованными запросами. По умолчанию оно выключено, чтобы автоматический запуск не ждал ввода. Для демонстрации включите его и выберите год или штат. Выход: 0.''')
code('''RUN_MENU = False
if RUN_MENU:
    run_menu(DATABASE)
else:
    print('Для демонстрации меню установите RUN_MENU=True и запустите эту ячейку.')
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
print('Ячеек',len(c),'SHA модулей',digest)
