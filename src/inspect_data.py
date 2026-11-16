"""Inspect a small CSV sample directly from the downloaded ZIP (stdlib only)."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile


def inspect_archive(path: Path, sample_size: int) -> dict:
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if m.filename.lower().endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected exactly one CSV in the archive')
        member = members[0]
        with archive.open(member) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
            columns = reader.fieldnames
            severity_counts = {}
            missing = {column: 0 for column in columns or []}
            count = 0
            for row in reader:
                count += 1
                severity = row.get('Severity', '')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                for column, value in row.items():
                    if value is None or value.strip() == '':
                        missing[column] = missing.get(column, 0) + 1
                if count >= sample_size:
                    break
    return {
        'archive': str(path.resolve()), 'csv': member.filename,
        'uncompressed_bytes': member.file_size, 'columns': columns,
        'sample_rows': count, 'sample_severity_counts': severity_counts,
        'sample_missing_counts': missing,
        'limitation': 'First rows only; not a random sample or a full quality audit.',
    }


def save_json(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('outputs/data_preview.json'))
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    try:
        result = inspect_archive(args.archive, args.rows)
        save_json(result, args.output)
    except (OSError, ValueError, zipfile.BadZipFile, csv.Error) as error:
        parser.exit(1, f'Data inspection failed: {error}\n')
    print(f"Inspected {result['sample_rows']} rows; {len(result['columns'])} columns")
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
