"""Единая точка запуска существующих модулей проекта."""
import argparse
import json
from pathlib import Path

from src.analysis import analyze_files
from src.database import create_database, run_queries
from src.menu import save_aggregates, run_menu
from src.visualization import create_figures, numeric_summary


def run_analysis(folder, database, output):
    """Повторно используем базу, считаем агрегаты, сохраняем SQL и графики."""
    folder, database, output = Path(folder),Path(database),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    results = analyze_files(folder)
    if not database.exists():
        create_database(folder,database)
    sql = run_queries(database,output/'sql')
    if int(sql['1_severity'].accidents.sum()) != int(results['severity'].accidents.sum()):
        raise ValueError('Количество строк SQLite и Parquet не совпало')
    save_aggregates(database,results,output/'aggregates')
    for name,table in results.items():
        table.to_csv(output/f'{name}_counts.csv',index=False)
    audit_file = folder/'audit.json'
    if not audit_file.exists():
        audit_file = output/'optimization_audit.json'
    audit = json.loads(audit_file.read_text())
    print('\n'.join(create_figures(results,audit,output/'figures')))
    if not (output/'numeric_description.csv').exists():
        numeric_summary(folder,output)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['analysis','models','benchmark','menu','audit','quality'])
    parser.add_argument('--folder',default='data/processed/full_parquet')
    parser.add_argument('--database',default='data/processed/accidents.db')
    parser.add_argument('--output',default='outputs')
    args=parser.parse_args()
    if args.stage=='analysis':
        run_analysis(args.folder,args.database,args.output)
    elif args.stage=='models':
        from src.final_modeling import run
        print(run(args.folder,Path(args.output)/'final').to_string(index=False))
    elif args.stage=='benchmark':
        from src.benchmark import compare_tools
        table,counts=compare_tools(args.folder)
        table.to_csv(Path(args.output)/'pandas_polars_benchmark.csv',index=False)
        print(table.to_string(index=False));print(counts)
    elif args.stage=='quality':
        from src.quality_hypotheses import analyze_quality,test_hypothesis
        quality,weather,day=analyze_quality(args.folder)
        weather.to_csv(Path(args.output)/'weather_severity.csv')
        day.to_csv(Path(args.output)/'day_severity.csv')
        report=dict(quality=quality,weather_test=test_hypothesis(weather),day_test=test_hypothesis(day))
        (Path(args.output)/'quality_hypotheses.json').write_text(json.dumps(report,indent=2))
    elif args.stage=='audit':
        from src.optimize import audit_and_optimize
        print(audit_and_optimize(args.folder,Path(args.folder).parent/'optimized_parquet',Path(args.output)/'optimization_audit.json'))
    else:
        run_menu(args.database)


if __name__=='__main__':
    main()
