"""Проверки контракта данных, меню и отсутствия пересечений без нового подбора."""
import contextlib
import io
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.menu import run_menu


def verify(root=Path('.')):
    out=root/'outputs'
    report=json.loads((out/'optimization_audit.json').read_text())
    assert report['rows']==7728394 and report['complete']
    assert report['checks']['duplicate_id_extra_rows']==0
    ids={}
    for name in ['train','validation','test']:
        frame=pd.read_parquet(out/f'final/splits/{name}.parquet',columns=['ID','Start_Time'])
        ids[name]=set(frame.ID)
        h=pd.util.hash_pandas_object(frame.ID.astype('string'),index=False)%10000
        assert (h.between(3000,3999) if name=='test' else h<3000).all()
        assert len(ids[name])==len(frame)
    assert not ids['train'] & ids['validation']
    assert not ids['train'] & ids['test']
    assert not ids['validation'] & ids['test']
    transcript=io.StringIO()
    with patch('builtins.input',side_effect=['9','1','bad','1','2021','2','ca','2','ZZ','0']), contextlib.redirect_stdout(transcript):
        run_menu(root/'data/processed/accidents.db')
    text=transcript.getvalue()
    assert 'Неизвестное действие' in text and 'Введите год' in text and 'Данные не найдены' in text and 'Выход' in text
    assert 'Severity' in text
    with sqlite3.connect(root/'data/processed/accidents.db') as db:
        for name in ['severity','state','year','hour','weather']:
            parquet=pd.read_parquet(out/f'aggregates/{name}.parquet')
            sql=pd.read_sql_query('SELECT * FROM agg_'+name,db)
            pd.testing.assert_frame_equal(parquet,sql,check_dtype=False)
            assert parquet.accidents.sum() <= 7728394
    (out/'verification.json').write_text(json.dumps({'full_audit_rows':report['rows'],'split_id_intersections':0,
        'test_hash_outside_old_samples':True,'menu_invalid_year_state_exit':True,'aggregate_parquet_sqlite_equal':True},indent=2))
    (out/'menu_demo.txt').write_text(text)
    print('Проверки полного аудита, разбиения, меню и агрегатов пройдены.')


if __name__=='__main__':
    verify()
