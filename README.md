# US Traffic Accidents Patterns and Severity Prediction

Группа **AAI-2602**: **Bekmukhamedova Meruert**, **Aiganym Gadilova**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MikaLinni/US-Traffic-Accidents/blob/main/US_Accidents_Colab.ipynb)

**Посмотреть без запуска:** [проверенные результаты](RESULTS.md),
[отчёт](docs/REPORT.md), [Word-версия](deliverables/US_Traffic_Accidents_Report.docx),
[презентация](deliverables/US_Traffic_Accidents_Presentation.pptx).

| Модель | Test F1-macro | Test Accuracy |
|---|---:|---:|
| Dummy | 0,2394 | 0,9184 |
| Logistic Regression | 0,1384 | 0,2236 |
| HGB с погодой | 0,2081 | 0,3984 |

Это зафиксированная временная проверка. HGB не превзошёл baseline.
Первый полный запуск скачивает CSV 3,06 ГБ и выполняет обработку и обучение.
Он занимает время, а не секунды. Фактическое выполнение на серверах Colab
пока не подтверждено. Готовые результаты можно проверить сразу по ссылкам выше.


Финальный проект №42 по Programming for Data Analysis and Database, Cluster 8.
Анализируем 7 728 394 зарегистрированных ДТП США за 2016–2023 и оцениваем
Severity — влияние ДТП на движение. Проект не оценивает тяжесть травм и
не предсказывает вероятность возникновения ДТП.

## Google Colab

1. Откройте `US_Accidents_Colab.ipynb` в Google Colab.
2. Выберите Runtime → Run all. Код модулей и данные загружаются автоматически. Пути менять и файлы загружать не нужно.
3. Ноутбук распаковывает обычный ZIP с настоящими `src/*.py`, устанавливает
   версии из requirements.txt и скачивает CSV с проверкой SHA-256.
4. При повторном запуске готовые Parquet, SQLite и метрики используются снова.
   После завершения создаётся `traffic_results.zip` с результатами.
5. Для меню включите `RUN_MENU=True` в соответствующей ячейке. Выход: `0`.

Скачивание занимает около 3,06 ГБ, исходный CSV и две версии Parquet вместе
с базой и результатами требуют нескольких ГБ диска. Время зависит от сети и CPU.
В `/content` файлы временные. Опциональный `USE_DRIVE=True` в начале ноутбука
сохраняет папку проекта между сеансами через Google Drive. Авторизация Drive
нужна только при включении этой опции. Ноутбук не удаляет исходные данные.

Проверено локальное выполнение всех ячеек в новом Python-процессе на полном
готовом наборе. Фактический запуск на серверах Google Colab пока не подтверждён.
Это различие отражено в `docs/CRITERIA_STATUS.md`.

## Локальный запуск из корня проекта

Python 3.13 использовался для проверенных результатов.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.pipeline analysis
python -m src.pipeline audit
python -m src.pipeline benchmark
python -m src.pipeline models
python -m src.pipeline menu
python scripts/execute_notebook.py
```

При наличии `data/processed/full_parquet` повторно читать CSV не нужно.
SQLite используется из `data/processed/accidents.db`; если её нет, стадия
analysis создаёт её из Parquet. Для новой установки без подготовленных данных
запустите ноутбук целиком локально: пути выбираются относительно текущей папки,
CSV загружается автоматически. Можно также подготовить архив своей командой:

```python
from src.prepare import prepare_data
prepare_data('/path/to/archive.zip', 'data/processed/full_parquet')
```

Не смешивайте в одной папке неполный пробный набор и полный набор.
Незавершённый этап ETL/downcast не считается кэшем: используйте новую папку
или вручную разберите причину сбоя. Отчёт audit.json создаётся после завершения.
Финальные метрики сохранены и повторно не подбираются. Новые исследовательские
варианты должны иметь отдельный протокол и папку результатов.

## Данные и источники

- Исходный набор: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- Публичная копия: https://huggingface.co/datasets/Sanjana7787/us-accidents-2016-2023
- Закреплённая версия CSV: commit `b514f1c`.
- SHA-256 CSV: `e3e9f962e3e2289db1bdce91623abbcf9ba4d7bfc1adb091c8aeee6c2639d204`.

CSV внутри пользовательского ZIP и CSV публичной копии имеют эту контрольную
сумму по предыдущей проверке. SHA относится к распакованному CSV, не к ZIP.
Большие данные и модели в Git не включаются. Применяйте условия использования
исходного набора, указанные его автором.

## Результаты и ограничения

Старый случайный validation: HGB F1-macro 0,615 на равных долях классов и
0,315 на естественных долях. Это не test.

Фиксированный временной test с другими ID (200 432 наблюдения):

| Модель | F1-macro, естественные доли | Accuracy |
|---|---:|---:|
| Dummy, всегда класс 2 | 0,2394 | 0,9184 |
| Logistic Regression | 0,1384 | 0,2236 |
| HGB с погодой | 0,2081 | 0,3984 |
| HGB без погоды | 0,2085 | 0,3985 |

HGB выбирался по validation до просмотра test, но на test baseline оказался
сильнее. Высокое качество переноса на будущие годы не достигнуто. Слабый
класс 1 не скрываем: F1 около 0,0275 у HGB. Подробные per-class таблицы,
confusion matrix, ошибки по погоде и время/RAM сохраняются в outputs/final.
Временной сдвиг и выбор обучающей подвыборки ограничивают результат.

`stratify` сохраняет доли классов. Веса меняют обучение. Оценка на равных
долях меняет состав проверяемой популяции. Финальные основные метрики
относятся к естественным долям. Повторов строк в финальном train нет.
Полные данные участвовали в EDA, поэтому test не полностью скрыт от автора.

## Структура и защита

- `src/prepare.py`, `optimize.py`: чанки, аудит, Parquet, типы и память.
- `src/database.py`, `analysis.py`, `benchmark.py`: SQL, агрегаты, lazy и сравнение.
- `src/final_modeling.py`: фиксированные разбиения, модели, ablation и ошибки.
- `src/menu.py`, `pipeline.py`: интерфейс и модульный запуск.
- `src/visualization.py`: пять графиков PNG, интерактивные Plotly HTML.
- `docs/FINAL_PROTOCOL.md`, `CLEANING.md`, `ER_DIAGRAM.md`: методы и ограничения.
- `docs/experiments/US_Accidents_Exploration.ipynb`: прежние исследования.
- `docs/REPORT.md`: письменный отчёт. `deliverables/`: презентация.
- `docs/DEFENSE.md`: объяснения и сценарий демонстрации.
- `scripts/build_notebook.py`: пересобирает notebook и project_code.zip после изменения src.
- `scripts/verify_project.py`: проверки полного набора, меню и агрегатов.

Погодный ablation и ошибки по погодным группам выделены как Original
Contribution / Additional Experiments. Это наши расширения. Требования
преподавателя и их проверенные статусы перечислены отдельно в
`docs/CRITERIA_STATUS.md`. Тему, состав команды и дату экзамена подтверждает
студент. Отчёт и презентацию нужно загрузить в Moodle за 2–3 дня до экзамена.
