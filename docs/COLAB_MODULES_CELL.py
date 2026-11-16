# Один раз выберите project_code.zip. Датасет вручную загружать не нужно.
archive = ROOT / 'project_code.zip'
if not (ROOT / 'src/pipeline.py').exists():
    if not archive.exists():
        from google.colab import files
        uploaded = files.upload()
        if 'project_code.zip' not in uploaded:
            raise FileNotFoundError('Выберите файл project_code.zip')
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (ROOT / name).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Недопустимый путь в ZIP')
        bundle.extractall(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

