"""Пять графиков и описательная статистика с сохранением результатов."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def create_figures(results, audit, output):
    """Сохраняем статические PNG и интерактивные HTML для защиты."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    descriptions = []
    for name, title, label in [
        ('severity','Влияние ДТП на движение по уровням Severity','Severity'),
        ('hour','ДТП по местному часу начала','Час'),
        ('year','ДТП по году начала','Год'),
        ('state','Десять штатов с наибольшим числом записей','Штат')]:
        data = results[name].copy()
        data[name] = data[name].astype(str)
        data = data.head(10) if name == 'state' else data.sort_values(name)
        fig,ax = plt.subplots(figsize=(9,4.5))
        ax.bar(data[name],data.accidents,color='#326f9c')
        ax.set(title=title,xlabel=label,ylabel='Количество зарегистрированных ДТП')
        ax.ticklabel_format(axis='y',style='plain')
        fig.tight_layout();fig.savefig(output/f'{name}.png',dpi=160);plt.close(fig)
        interactive=px.bar(data,x=name,y='accidents',title=title,labels={name:label,'accidents':'Число ДТП'})
        interactive.write_html(output/f'{name}.html',include_plotlyjs=True)
        top = data.loc[data.accidents.idxmax()]
        descriptions.append(f'{title}: максимум в категории {top[name]} — {int(top.accidents):,} записей.')
    missing = pd.Series(audit['missing']).sort_values(ascending=False).head(10)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(missing.index[::-1],missing.values[::-1],color='#326f9c')
    ax.set(title='Десять признаков с наибольшим числом пропусков',xlabel='Число пропусков')
    fig.tight_layout();fig.savefig(output/'missing.png',dpi=160);plt.close(fig)
    descriptions += ['Количество записей по годам отражает также изменение покрытия источников; 2023 год неполный.',
                     'Число ДТП по штатам не является риском: нет знаменателя по пробегу, населению и интенсивности движения.',
                     'Пропуски конечных координат и Wind_Chill не заполняем для EDA; эти поля не нужны итоговой модели.']
    (output/'interpretation.txt').write_text('\n'.join(descriptions),encoding='utf-8')
    return descriptions


def numeric_summary(folder, output):
    """Описательная статистика чисел через Polars lazy по всему набору."""
    import polars as pl
    from src.optimize import NUMBERS
    files=sorted(str(p) for p in Path(folder).glob('year=*/part-*.parquet'))
    scan=pl.scan_parquet(files,hive_partitioning=False)
    rows=[]
    for name in NUMBERS:
        x=pl.col(name).cast(pl.Float64,strict=False)
        values=scan.select(x.count().alias('count'),x.mean().alias('mean'),x.std().alias('std'),
                           x.min().alias('min'),x.quantile(0.25).alias('q25'),
                           x.median().alias('median'),x.quantile(0.75).alias('q75'),x.max().alias('max')).collect(engine='streaming').to_dicts()[0]
        rows.append(dict(feature=name,**values))
    table=pd.DataFrame(rows)
    table.to_csv(Path(output)/'numeric_description.csv',index=False)
    return table
