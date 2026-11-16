"""Выполняет итоговый ноутбук в новом Python-процессе без старых переменных."""
import json
import os
import sys
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

root=Path(__file__).resolve().parents[1]
os.chdir(root)
notebook=nbformat.read(root/'US_Accidents_Colab.ipynb',as_version=4)
shell=InteractiveShell.instance()
for index,cell in enumerate(notebook.cells):
    if cell.cell_type=='code':
        print('Ячейка',index,flush=True)
        result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Ошибка ячейки {index}')
(root/'outputs/notebook_execution.json').write_text(json.dumps({'success':True,'cells':len(notebook.cells),
    'environment':'fresh local Python process, cached full Parquet and frozen model results',
    'colab_verified':False,'fresh_download_verified':False},indent=2))
