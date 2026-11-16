"""SQLite: загрузка записей ДТП и аналитические запросы."""
import sqlite3
from pathlib import Path
import pandas as pd


def create_database(parquet_folder, database_path):
    """Создаёт базу и загружает данные из Parquet по одному файлу."""
    files = sorted(Path(parquet_folder).glob('year=*/part-*.parquet'))
    if not files:
        raise FileNotFoundError('Файлы Parquet не найдены')
    database_path = Path(database_path)
    if database_path.exists():
        raise FileExistsError(f'База уже существует: {database_path}')
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute('PRAGMA foreign_keys = ON')
        connection.executescript('''
            CREATE TABLE states (
                state_id INTEGER PRIMARY KEY,
                state_code TEXT NOT NULL UNIQUE
            );
            CREATE TABLE accidents (
                accident_id INTEGER PRIMARY KEY,
                state_id INTEGER,
                severity INTEGER,
                start_time TEXT,
                weather TEXT,
                FOREIGN KEY (state_id) REFERENCES states(state_id)
            );
            CREATE TABLE project_info (row_count INTEGER NOT NULL);
        ''')
        # Справочник штатов избегает многократного хранения названия штата.
        state_ids = {}
        columns = ['State', 'Severity', 'Start_Time', 'Weather_Condition']
        for number, file in enumerate(files, 1):
            frame = pd.read_parquet(file, columns=columns)
            for state in frame['State'].dropna().unique():
                if state not in state_ids:
                    cursor = connection.execute('INSERT INTO states (state_code) VALUES (?)', (state,))
                    state_ids[state] = cursor.lastrowid
            state_column = frame['State'].map(state_ids)
            severity = pd.to_numeric(frame['Severity'], errors='coerce')
            rows = zip(state_column, severity, frame['Start_Time'], frame['Weather_Condition'])
            # Превращаем пропуски в None, чтобы SQLite сохранил их как NULL.
            batch = []
            for state, level, start, weather in rows:
                batch.append((None if pd.isna(state) else int(state),
                              None if pd.isna(level) else int(level),
                              None if pd.isna(start) else str(start),
                              None if pd.isna(weather) else str(weather)))
            connection.executemany(
                'INSERT INTO accidents (state_id, severity, start_time, weather) VALUES (?, ?, ?, ?)', batch)
            connection.commit()
            if number % 20 == 0 or number == len(files):
                print(f'Загружено файлов: {number} из {len(files)}')
        connection.execute('CREATE INDEX idx_accidents_state ON accidents(state_id)')
        connection.execute('INSERT INTO project_info SELECT COUNT(*) FROM accidents')
    return database_path


# Ключи описывают смысл запросов. Запросы используют одинаковые исходные данные.
QUERIES = {
    '1_severity': '''
        SELECT severity, COUNT(*) AS accidents
        FROM accidents GROUP BY severity ORDER BY severity
    ''',
    '2_states_join': '''
        SELECT s.state_code, COUNT(*) AS accidents
        FROM accidents a JOIN states s ON a.state_id = s.state_id
        GROUP BY s.state_code ORDER BY accidents DESC LIMIT 10
    ''',
    '3_states_having': '''
        SELECT s.state_code, COUNT(*) AS accidents,
               ROUND(AVG(a.severity), 2) AS average_severity
        FROM accidents a JOIN states s ON a.state_id = s.state_id
        GROUP BY s.state_code HAVING COUNT(*) >= 10000
        ORDER BY average_severity DESC
    ''',
    '4_hour': '''
        SELECT SUBSTR(start_time, 12, 2) AS hour, COUNT(*) AS accidents
        FROM accidents WHERE start_time IS NOT NULL
        GROUP BY hour ORDER BY hour
    ''',
    '5_weather': '''
        SELECT weather, COUNT(*) AS accidents
        FROM accidents WHERE weather IS NOT NULL
        GROUP BY weather ORDER BY accidents DESC LIMIT 10
    ''',
    '6_state_above_average': '''
        WITH state_counts AS (
            SELECT s.state_code, COUNT(*) AS accidents
            FROM accidents a JOIN states s ON a.state_id = s.state_id
            GROUP BY s.state_code
        )
        SELECT state_code, accidents FROM state_counts
        WHERE accidents > (SELECT AVG(accidents) FROM state_counts)
        ORDER BY accidents DESC
    ''',
    '7_state_rank': '''
        WITH state_counts AS (
            SELECT s.state_code, COUNT(*) AS accidents
            FROM accidents a JOIN states s ON a.state_id = s.state_id
            GROUP BY s.state_code
        )
        SELECT state_code, accidents,
               RANK() OVER (ORDER BY accidents DESC) AS state_rank
        FROM state_counts ORDER BY state_rank
    ''',
    '8_year_severity': '''
        SELECT SUBSTR(start_time, 1, 4) AS year, severity, COUNT(*) AS accidents
        FROM accidents WHERE start_time IS NOT NULL
        GROUP BY year, severity ORDER BY year, severity
    '''
}


def run_queries(database_path, output_folder):
    """Выполняет запросы, возвращает таблицы Pandas и сохраняет их в CSV."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    results = {}
    with sqlite3.connect(database_path) as connection:
        for name, query in QUERIES.items():
            table = pd.read_sql_query(query, connection)
            table.to_csv(output_folder / f'{name}.csv', index=False)
            results[name] = table
    return results
