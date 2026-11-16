"""Простое меню для просмотра данных SQLite с фильтрами."""
import sqlite3
from pathlib import Path
import pandas as pd


def save_aggregates(database_path, results, output_folder):
    """Сохраняем уже рассчитанные агрегаты в SQLite и Parquet."""
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        for name in ['severity', 'state', 'year', 'hour', 'weather']:
            table = results[name]
            table.to_sql('agg_' + name, connection, if_exists='replace', index=False)
            table.to_parquet(folder / (name + '.parquet'), index=False)
    print('Агрегаты сохранены в SQLite и Parquet.')


def run_menu(database_path):
    """Показывает статистику по выбранному году или штату."""
    try:
        with sqlite3.connect(Path(database_path).resolve().as_uri() + '?mode=ro', uri=True) as connection:
            while True:
                print('\n1 — ДТП по Severity за год')
                print('2 — ДТП по Severity в штате')
                print('0 — выход')
                choice = input('Выберите действие: ').strip()
                if choice == '0':
                    print('Выход из меню.')
                    break
                elif choice == '1':
                    year = input('Введите год, например 2021: ').strip()
                    if not (year.isdigit() and len(year) == 4):
                        print('Введите год из четырёх цифр.')
                        continue
                    query = '''SELECT severity AS Severity, COUNT(*) AS accidents
                               FROM accidents WHERE SUBSTR(start_time, 1, 4) = ?
                               GROUP BY severity ORDER BY severity'''
                    result = pd.read_sql_query(query, connection, params=(year,))
                elif choice == '2':
                    state = input('Введите код штата, например CA: ').strip().upper()
                    query = '''SELECT a.severity AS Severity, COUNT(*) AS accidents
                               FROM accidents a JOIN states s ON a.state_id = s.state_id
                               WHERE s.state_code = ? GROUP BY a.severity ORDER BY a.severity'''
                    result = pd.read_sql_query(query, connection, params=(state,))
                else:
                    print('Неизвестное действие. Выберите 0, 1 или 2.')
                    continue
                if result.empty:
                    print('Данные не найдены.')
                else:
                    print(result.to_string(index=False))
    except (sqlite3.Error, OSError) as error:
        print('Не удалось открыть базу:', error)
