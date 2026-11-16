"""Интерактивная панель Colab для анализа сохранённых данных SQLite."""
import html
from pathlib import Path
import sqlite3

import pandas as pd


def dashboard_query(database_path, year=None, state=None, severity=None):
    """Одна параметризованная SQL-агрегация для выбранных фильтров."""
    conditions, parameters = [], []
    if year is not None:
        year = int(year)
        conditions += ['a.start_time >= ?', 'a.start_time < ?']
        parameters += [f'{year}-01-01', f'{year+1}-01-01']
    if state is not None:
        conditions.append('s.state_code = ?')
        parameters.append(str(state))
    if severity is not None:
        severity = int(severity)
        if severity not in [1, 2, 3, 4]:
            raise ValueError('Severity должен быть от 1 до 4')
        conditions.append('a.severity = ?')
        parameters.append(severity)
    where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
    query = '''SELECT s.state_code AS state, a.severity AS Severity,
                      SUBSTR(a.start_time, 12, 2) AS hour,
                      COALESCE(a.weather, 'Unknown') AS weather,
                      COUNT(*) AS accidents
               FROM accidents a LEFT JOIN states s ON a.state_id = s.state_id'''
    query += where + ' GROUP BY s.state_code, a.severity, SUBSTR(a.start_time, 12, 2), a.weather'
    uri = Path(database_path).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as connection:
        return pd.read_sql_query(query, connection, params=parameters)


def show_dashboard(database_path, output_folder='outputs/dashboard'):
    """Показываем фильтры, показатели, таблицу, графики и сохранение выборки."""
    import ipywidgets as widgets
    import plotly.express as px
    from IPython.display import HTML, clear_output, display

    database_path = Path(database_path)
    if not database_path.is_file():
        raise FileNotFoundError('Сначала выполните раздел создания SQLite.')
    uri = database_path.resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as connection:
        states = pd.read_sql_query('SELECT state_code FROM states ORDER BY state_code', connection).state_code.tolist()
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'agg_year' in tables:
            years = pd.read_sql_query('SELECT year FROM agg_year ORDER BY year', connection).year.astype(str).tolist()
        else:
            years = pd.read_sql_query('SELECT DISTINCT SUBSTR(start_time,1,4) AS year FROM accidents WHERE start_time IS NOT NULL ORDER BY year', connection).year.tolist()
    years = sorted({int(y) for y in years if str(y).isdigit() and len(str(y)) == 4})
    style = {'description_width': 'initial'}
    year_widget = widgets.Dropdown(options=[('Все годы', None)]+[(str(y),y) for y in years],
                                   value=2020 if 2020 in years else None, description='Год:', style=style)
    state_widget = widgets.Dropdown(options=[('Все штаты',None)]+[(s,s) for s in states],
                                    description='Штат:', style=style)
    severity_widget = widgets.Dropdown(options=[('Все уровни',None)]+[(str(x),x) for x in [1,2,3,4]],
                                       description='Severity:', style=style)
    update = widgets.Button(description='Показать результаты', button_style='primary', icon='bar-chart',
                            layout=widgets.Layout(width='215px'))
    save = widgets.Button(description='Сохранить CSV', icon='download', disabled=True,
                          layout=widgets.Layout(width='175px'))
    status = widgets.HTML()
    panel = widgets.Output()
    current = {}
    display(HTML('''<div style="font-family:Arial;padding:18px 0">
        <h2 style="margin:0;color:#183145">US Traffic Accidents</h2>
        <p style="color:#526475">Выберите период, штат и уровень влияния ДТП на движение.</p>
        </div>'''))
    display(widgets.VBox([widgets.HBox([year_widget,state_widget,severity_widget],
                                      layout=widgets.Layout(flex_flow='row wrap')),
                          widgets.HBox([update,save]),status,panel]))

    def render(_=None):
        update.disabled = True
        save.disabled = True
        status.value = 'Выполняется запрос к SQLite…'
        with panel:
            clear_output(wait=True)
            try:
                frame = dashboard_query(database_path,year_widget.value,state_widget.value,severity_widget.value)
                chosen = dict(year=year_widget.value,state=state_widget.value,severity=severity_widget.value)
                current.update(frame=frame, filters=chosen)
                if frame.empty:
                    status.value = 'Для выбранных фильтров записей нет.'
                    return
                count = int(frame.accidents.sum())
                severe = int(frame.loc[frame.Severity.isin([3,4]),'accidents'].sum())
                period = str(chosen['year']) if chosen['year'] else '2016–2023'
                region = chosen['state'] or 'Все штаты'
                severity_text = str(chosen['severity']) if chosen['severity'] else 'все'
                display(HTML(f'''<h3>{html.escape(region)} · {period} · Severity: {severity_text}</h3>
                    <div style="display:flex;gap:45px;flex-wrap:wrap;padding:12px 0">
                    <div><span style="color:#526475">Зарегистрировано ДТП</span><br>
                    <strong style="font-size:30px;color:#183145">{count:,}</strong></div>
                    <div><span style="color:#526475">Доля Severity 3–4 в выбранных данных</span><br>
                    <strong style="font-size:30px;color:#183145">{severe/count:.1%}</strong></div></div>'''))
                levels = frame.groupby('Severity',as_index=False).accidents.sum().sort_values('Severity')
                levels['share_percent'] = (levels.accidents/count*100).round(2)
                display(levels.rename(columns={'accidents':'Количество ДТП','share_percent':'Доля, %'}).reset_index(drop=True))
                fig = px.bar(levels,x='Severity',y='accidents',color='Severity',
                             color_discrete_sequence=['#326f9c'],title='Распределение Severity',
                             labels={'accidents':'Число ДТП'})
                fig.update_xaxes(type='category');fig.update_layout(showlegend=False)
                fig.show()
                hours = frame.groupby('hour',as_index=False).accidents.sum().sort_values('hour')
                px.bar(hours,x='hour',y='accidents',title='ДТП по местному часу начала',
                       labels={'hour':'Местный час','accidents':'Число ДТП'},
                       color_discrete_sequence=['#2a9d8f']).show()
                weather = frame.groupby('weather',as_index=False).accidents.sum().nlargest(10,'accidents').sort_values('accidents')
                px.bar(weather,x='accidents',y='weather',orientation='h',title='Частые погодные условия',
                       labels={'weather':'Погода','accidents':'Число ДТП'},color_discrete_sequence=['#8064a2']).show()
                if chosen['state'] is None:
                    top_states = frame.groupby('state',as_index=False).accidents.sum().nlargest(10,'accidents')
                    px.bar(top_states,x='state',y='accidents',title='Десять штатов по числу записей',
                           labels={'state':'Штат','accidents':'Число ДТП'}).show()
                display(HTML('<p style="color:#526475">Это количество зарегистрированных ДТП, а не риск поездки. '
                             'Severity характеризует влияние на движение. Данные 2023 года неполные. '
                             'Погода здесь используется для описания исходных событий, включая измерения после начала; '
                             'в прогнозной модели доступность погоды проверяется отдельно.</p>'))
                status.value = 'Показаны результаты выбранных фильтров.'
                save.disabled = False
            except (sqlite3.Error, OSError, ValueError) as error:
                status.value = 'Не удалось получить данные: ' + html.escape(str(error))
            finally:
                update.disabled = False

    def save_result(_):
        if 'frame' not in current:
            return
        folder = Path(output_folder)
        folder.mkdir(parents=True,exist_ok=True)
        chosen = current['filters']
        path = folder / f"accidents_{chosen['year'] or 'all'}_{chosen['state'] or 'all'}_severity_{chosen['severity'] or 'all'}.csv"
        current['frame'].to_csv(path,index=False)
        status.value = 'Сохранены агрегаты для показанных фильтров: ' + html.escape(str(path))
        try:
            from google.colab import files
        except ImportError:
            return
        files.download(str(path))

    update.on_click(render)
    save.on_click(save_result)
    render()
    return panel
