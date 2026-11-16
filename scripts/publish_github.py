"""Публикация подготовленного пакета из Colab в новый GitHub-репозиторий."""
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


def github_api(token, endpoint, payload=None):
    """Токен передаём в заголовке, а не в URL или выводе ячейки."""
    request = urllib.request.Request(
        'https://api.github.com' + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + token,
                 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2026-03-10',
                 'User-Agent': 'US-Accidents-Course-Project'},
        method='GET' if payload is None else 'POST')
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Не печатаем запросы и заголовки авторизации.
        if error.code == 422:
            raise RuntimeError('Репозиторий с таким именем уже существует или имя недопустимо. Выберите новое имя. Существующий репозиторий не изменён.') from None
        raise RuntimeError(f'GitHub HTTP {error.code}. Проверьте срок действия и разрешения токена.') from None


def git(args, cwd, env=None):
    """Запускаем Git без shell и без секретов в аргументах команды."""
    result = subprocess.run(['git', '-c', 'credential.helper='] + args,
                            cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Команда Git не завершилась: ' + ' '.join(args[:2]) +
                           '. Проверьте доступ к репозиторию и сеть.')
    return result.stdout


def configure_public_project(folder, login, name):
    """Добавляем настоящие ссылки README и автоматическое получение модулей."""
    folder = Path(folder)
    repo = f'{login}/{name}'
    url = f'https://github.com/{repo}'
    colab = f'https://colab.research.google.com/github/{repo}/blob/main/US_Accidents_Colab.ipynb'
    path = folder / 'US_Accidents_Colab.ipynb'
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if 'archive = ROOT /' in source and 'project_code.zip' in source:
            cell['source'] = (f'''# Получаем обычный ZIP модулей непосредственно из этого репозитория.
import urllib.request
archive = ROOT / 'project_code.zip'
if not (ROOT / 'src/pipeline.py').exists():
    if not archive.exists():
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/{repo}/main/deliverables/project_code.zip',
            archive)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (ROOT / name).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Недопустимый путь в ZIP')
        bundle.extractall(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
''').splitlines(keepends=True)
    intro = ''.join(notebook['cells'][0]['source'])
    intro = intro.replace('Для Colab нужны этот ноутбук и небольшой `project_code.zip`: выберите ZIP один раз в ячейке загрузки модулей.',
                          'В этом публичном репозитории код модулей загружается автоматически: ручная загрузка ZIP не требуется.')
    notebook['cells'][0]['source'] = intro.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    readme = folder / 'README.md'
    text = readme.read_text()
    text = text.replace('2. Скачайте также `deliverables/project_code.zip`. Выберите Runtime → Run all;\n   в ячейке модулей один раз выберите этот ZIP. Пути менять не нужно.',
                        '2. Выберите Runtime → Run all. Код модулей и данные загружаются автоматически. Пути менять и файлы загружать не нужно.')
    quick = f'''Группа **AAI-2602**: **Bekmukhamedova Meruert**, **Aiganym Gadilova**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})

**Посмотреть без запуска:** [проверенные результаты](RESULTS.md),
[отчёт](docs/REPORT.md), [Word-версия](deliverables/US_Traffic_Accidents_Report.docx),
[презентация](deliverables/US_Traffic_Accidents_Presentation.pptx).

| Модель | Test F1-macro | Test Accuracy |
|---|---:|---:|
| Dummy | 0,2394 | 0,9184 |
| Logistic Regression | 0,1384 | 0,2236 |
| HGB с погодой | 0,2081 | 0,3984 |

Это зафиксированная временная проверка. HGB не превзошёл baseline.
Первый полный запуск скачивает CSV 3,06 ГБ и выполняет обработку и обучение.
Он занимает время, а не секунды. Фактическое выполнение на серверах Colab
пока не подтверждено. Готовые результаты можно проверить сразу по ссылкам выше.

'''
    heading, rest = text.split('\n', 1)
    readme.write_text(heading + '\n\n' + quick + rest)
    return url, colab


def main():
    package = Path(__file__).resolve().parent
    if not (package/'project/README.md').exists():
        raise FileNotFoundError('Запускайте этот файл из распакованного github_ready.zip')
    print('Будет создан НОВЫЙ ПУБЛИЧНЫЙ репозиторий. Большие данные и модели не загружаются.')
    name = input('Название репозитория [US-Traffic-Accidents]: ').strip() or 'US-Traffic-Accidents'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', name):
        raise ValueError('Используйте латинские буквы, цифры, точку, дефис или подчёркивание.')
    token = getpass.getpass('GitHub token (ввод скрыт): ').strip()
    try:
        account = github_api(token, '/user')
        login = account['login']
        with tempfile.TemporaryDirectory(prefix='traffic_publish_') as temporary:
            work = Path(temporary)/'repository'
            history = package/'history.bundle'
            if history.exists():
                git(['clone', str(history), str(work)], package)
            else:
                work.mkdir()
                git(['init', '-b', 'main'], work)
            shutil.copytree(package/'project',work,dirs_exist_ok=True)
            url, colab = configure_public_project(work,login,name)
            git(['config','user.name',account.get('name') or login],work)
            git(['config','user.email',f'{account["id"]}+{login}@users.noreply.github.com'],work)
            git(['add','.'],work)
            if git(['status','--porcelain'],work).strip():
                git(['commit','-m','Prepare final project with results and automatic Colab launch'],work)
            # Создаём репозиторий только после полной подготовки локальных файлов.
            github_api(token,'/user/repos',{'name':name,'private':False,'auto_init':False,
                       'description':'AAI-2602 final project: US Traffic Accidents, analysis and severity prediction'})
            askpass = Path(temporary)/'askpass.py'
            askpass.write_text('#!' + sys.executable + '\nimport os, sys\nprint("x-access-token" if "username" in sys.argv[1].lower() else os.environ["TRAFFIC_GH_TOKEN"])\n')
            askpass.chmod(0o700)
            env = dict(os.environ,GIT_ASKPASS=str(askpass),GIT_TERMINAL_PROMPT='0',TRAFFIC_GH_TOKEN=token)
            try:
                git(['remote','remove','origin'],work)
                git(['remote','add','origin',url+'.git'],work)
                git(['push','--set-upstream','origin','main'],work,env)
            finally:
                env.pop('TRAFFIC_GH_TOKEN',None)
        print('Проект опубликован:',url)
        print('Запуск:',colab)
        print('Токен не записан в файлы проекта. После публикации его можно отозвать в GitHub.')
    finally:
        token = ''


if __name__ == '__main__':
    main()
