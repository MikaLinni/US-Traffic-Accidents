# Числовой EDA: полная статистика и ограниченная выборка для графиков.
from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import display, Image, Markdown


def show_plot():
    buffer = io.BytesIO()
    plt.gcf().savefig(buffer, format='png', dpi=140, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
    plt.close()


def run_numeric_eda(parquet, output, total_rows):
    """Показывает числовой EDA и сохраняет результаты в output/numeric_eda."""
    # Эта таблица рассчитана ранее по ВСЕМ записям, а не по выборке.
    OUT = Path(output)
    PARQUET = Path(parquet)
    audit = {'rows': total_rows}
    full_stats = pd.read_csv(OUT / 'numeric_description.csv')
    full_stats['missing_percent'] = (1 - full_stats['count'] / audit['rows']) * 100
    full_stats['IQR'] = full_stats['q75'] - full_stats['q25']
    display(Markdown('### Числовые признаки: полный набор'))
    display(full_stats.round(3))

    # Для графиков используем только старую обучающую область ID и годы до 2021.
    # Новые наблюдения test не используем для дополнительных решений по модели.
    # Детерминированный отбор около 1% строк; в памяти хранится только выборка.
    features = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
                'Wind_Speed(mph)', 'Pressure(in)', 'Precipitation(in)', 'Distance(mi)']
    parts = []
    for path in sorted(Path(PARQUET).glob('year=*/part-*.parquet')):
        if int(path.parent.name.split('=')[1]) >= 2021:
            continue
        chunk = pd.read_parquet(path, columns=['ID', 'Severity'] + features)
        bucket = pd.util.hash_pandas_object(chunk.ID.astype('string'), index=False) % 10000
        parts.append(chunk.loc[bucket < 100, ['Severity'] + features])
    if not parts:
        raise ValueError('Нет Parquet-файлов за годы до 2021.')
    eda = pd.concat(parts, ignore_index=True)
    if eda.empty:
        raise ValueError('Выборка EDA пуста.')
    eda['Severity'] = pd.to_numeric(eda['Severity'], errors='coerce')
    for name in features:
        eda[name] = pd.to_numeric(eda[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
    eda_folder = OUT / 'numeric_eda'
    eda_folder.mkdir(parents=True, exist_ok=True)

    # Настоящий вызов Pandas describe, отдельно от статистики полного набора.
    display(Markdown('### describe() на выборке'))
    description = eda[features].describe(percentiles=[.01, .25, .5, .75, .99]).T
    description.to_csv(eda_folder / 'sample_describe.csv')
    display(description.round(3))

    # IQR обозначает необычные значения, а не автоматически ошибки данных.
    q1, q3 = eda[features].quantile(.25), eda[features].quantile(.75)
    iqr = q3 - q1
    outside = eda[features].lt(q1 - 1.5 * iqr) | eda[features].gt(q3 + 1.5 * iqr)
    outliers = pd.DataFrame({'IQR': iqr, 'lower': q1 - 1.5 * iqr,
                             'upper': q3 + 1.5 * iqr, 'outside_count': outside.sum(),
                             'observed_count': eda[features].count()})
    outliers['outside_percent'] = 100 * outliers.outside_count / outliers.observed_count
    outliers.to_csv(eda_folder / 'sample_iqr.csv')
    display(Markdown('### Выбросы по правилу 1,5 × IQR'))
    display(outliers.round(3))

    # Сохраняем исходные значения; масштаб графиков ограничиваем только для читаемости.
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, name in zip(axes.flat, features):
        values = eda[name].dropna()
        low, high = values.quantile([.01, .99])
        visible = values.between(low, high)
        ax.hist(values[visible], bins=35, color='#2563eb', alpha=.8)
        ax.set(title=name, ylabel='Число записей')
        ax.text(.98, .95, f'Вне диапазона: {(~visible).sum():,}',
                ha='right', va='top', transform=ax.transAxes, fontsize=8)
    for ax in list(axes.flat)[len(features):]:
        ax.set_visible(False)
    fig.suptitle('Распределения числовых признаков: диапазон 1–99 перцентилей')
    fig.tight_layout()
    fig.savefig(eda_folder / 'distributions.png', dpi=140)
    show_plot()

    # Корреляция Спирмена описывает монотонные связи, не причинность.
    correlation = eda[features].corr(method='spearman', min_periods=100)
    pair_counts = eda[features].notna().astype('int64').T.dot(eda[features].notna().astype('int64'))
    correlation.to_csv(eda_folder / 'spearman.csv')
    pair_counts.to_csv(eda_folder / 'correlation_pair_counts.csv')
    display(Markdown('### Связи между числовыми признаками'))
    fig = px.imshow(correlation, zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                    text_auto='.2f', title='Корреляция Спирмена · выборка до 2021')
    fig.write_html(eda_folder / 'correlation.html')
    fig.show()
    display(pair_counts)

    # Severity — категориальная цель: сравниваем распределения по группам.
    by_class = eda.groupby('Severity')[features].agg(['count', 'median', 'mean'])
    by_class.to_csv(eda_folder / 'by_severity.csv')
    display(Markdown('### Числовые признаки по Severity'))
    display(by_class.round(2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)']):
        groups = [eda.loc[eda.Severity == level, name].dropna() for level in [1, 2, 3, 4]]
        ax.boxplot(groups, tick_labels=['1', '2', '3', '4'], showfliers=False)
        ax.set(title=name, xlabel='Severity')
    fig.suptitle('Распределения по Severity')
    fig.tight_layout()
    fig.savefig(eda_folder / 'severity_boxplots.png', dpi=140)
    show_plot()

