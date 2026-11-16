"""Собирает небольшой пакет для автоматической публикации из Colab."""
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def build():
    files=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*.py'))
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'src').glob('*.py'))
    files.update(['deliverables/project_code.zip','docs/COLAB_MODULES_CELL.py'])
    files.discard('deliverables/github_ready.zip')
    with tempfile.TemporaryDirectory() as tmp:
        history=Path(tmp)/'history.bundle'
        subprocess.run(['git','bundle','create',str(history),'--all'],cwd=ROOT,check=True,capture_output=True)
        target=ROOT/'deliverables/github_ready.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for name in sorted(files):
                path=ROOT/name
                if path.is_file():
                    if name.startswith(('data/','.git/','.venv/','.build/')) or path.suffix in ['.joblib','.db']:
                        raise ValueError('В пакет попали большие данные или модель: '+name)
                    z.write(path,'project/'+name)
            z.write(ROOT/'scripts/publish_github.py','publish_github.py')
            z.write(history,'history.bundle')
            z.writestr('project/RESULTS.md',results_page())
        print(target,'bytes',target.stat().st_size)
        return target

def results_page():
    return '''# Проверенные результаты проекта

AAI-2602. Bekmukhamedova Meruert, Aiganym Gadilova.
Это сохранённые результаты локальных запусков, а не утверждение о запуске в Colab.

## Графики полного набора

![Severity](outputs/figures/severity.png)
![Местный час](outputs/figures/hour.png)
![Годы](outputs/figures/year.png)
![Штаты](outputs/figures/state.png)
![Пропуски](outputs/figures/missing.png)

## Финальная временная оценка

| Модель | F1-macro | Accuracy |
|---|---:|---:|
| Dummy | 0,2394 | 0,9184 |
| Logistic Regression | 0,1384 | 0,2236 |
| HGB с погодой | 0,2081 | 0,3984 |
| HGB без погоды | 0,2085 | 0,3985 |

Test: 200 432 записи, естественные доли. Выбор HGB сделан по validation до test.
HGB не превзошёл baseline. Метрики слабых классов не исключены.
Старые 0,615/0,315 относятся к другому validation, не test.

[Подробный протокол](docs/FINAL_PROTOCOL.md),
[все классы HGB](outputs/final/hgb_none_test_classes.csv),
[матрица ошибок](outputs/final/hgb_none_test_confusion.csv),
[ошибки по погоде](outputs/final/hgb_none_test_weather_errors.csv),
[время и память моделей](outputs/final/validation_scores.csv),
[аудит и downcast](outputs/optimization_audit.json),
[проверка требований](docs/CRITERIA_STATUS.md), [отчёт](docs/REPORT.md).

GitHub показывает готовые таблицы и картинки без вычислений. Для интерактивных
графиков и воспроизведения используется основной ноутбук через кнопку Colab.
'''

if __name__=='__main__':
    build()
