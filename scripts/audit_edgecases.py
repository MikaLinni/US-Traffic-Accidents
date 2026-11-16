"""Дополнительные флаги необычных значений для ручной проверки."""
import json
from pathlib import Path
import polars as pl


def audit_edgecases(folder='data/processed/full_parquet',output='outputs/quality_edgecases.json'):
    files=sorted(str(p) for p in Path(folder).glob('year=*/part-*.parquet'))
    checks={
        'wind_over_200_mph':pl.col('Wind_Speed(mph)').cast(pl.Float64,strict=False)>200,
        'pressure_outside_20_35_in':~pl.col('Pressure(in)').cast(pl.Float64,strict=False).is_between(20,35),
        'visibility_over_100_mi':pl.col('Visibility(mi)').cast(pl.Float64,strict=False)>100,
        'precipitation_over_10_in':pl.col('Precipitation(in)').cast(pl.Float64,strict=False)>10,
        'distance_over_100_mi':pl.col('Distance(mi)').cast(pl.Float64,strict=False)>100}
    result=pl.scan_parquet(files,hive_partitioning=False).select([v.sum().alias(k) for k,v in checks.items()]).collect(engine='streaming').to_dicts()[0]
    result['note']='Пороговые флаги для ручной проверки, не доказанные ошибки. Эти признаки не используются финальной моделью; исходные значения не удаляем.'
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    print(audit_edgecases())
