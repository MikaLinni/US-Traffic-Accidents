"""Проверяемая автоматическая загрузка публичной копии набора."""
import hashlib
import urllib.request
from pathlib import Path

URL = 'https://huggingface.co/datasets/Sanjana7787/us-accidents-2016-2023/resolve/b514f1c/US_Accidents_March23.csv'
EXPECTED_SHA256 = 'e3e9f962e3e2289db1bdce91623abbcf9ba4d7bfc1adb091c8aeee6c2639d204'


def sha256_file(path):
    """Хешируем частями, не загружая весь файл в память."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(path):
    """Не заменяем существующий файл с неправильной контрольной суммой."""
    path = Path(path)
    if path.exists():
        if sha256_file(path) != EXPECTED_SHA256:
            raise ValueError('SHA-256 существующего файла не совпадает')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.download')
    try:
        request=urllib.request.Request(URL,headers={'User-Agent':'US-Accidents-course-project'})
        with urllib.request.urlopen(request,timeout=60) as response, temporary.open('wb') as target:
            for block in iter(lambda:response.read(8*1024*1024),b''):
                target.write(block)
        if sha256_file(temporary) != EXPECTED_SHA256:
            raise ValueError('SHA-256 загруженного файла не совпадает')
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path
