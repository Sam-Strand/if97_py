'''
Размен для области 3: скалярная скорость против холодного старта.

Замораживание таблиц в замыкании убирает распаковку пяти массивов и ускоряет
скалярный вызов вдвое. Но numba не кэширует функции, которые держат массив
замороженным: адрес данных вшивается в код, и кэш отключается
(`has_dynamic_globals`). Здесь измеряется, сколько это стоит.

Запуск:
    python tests/exp_cache.py
'''

import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARGS = '''
from if97.fluid import subregion, tables
A = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)
subregion.v_pt(650.0, 30.0, *A)
'''

CLOSED = '''
import numpy as np
from numba import njit
from if97._jit import JIT
from if97.fluid import subregion, tables

OFF, PAR, VN, LI, LJ = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)
v_reg, of_tp, scratch = subregion.v_reg, subregion.of_tp, subregion.scratch

@njit(cache=True, fastmath=True, nogil=True, error_model='numpy')
def v_closed(t, p):
    pw_x, pw_y = scratch()
    return v_reg(t, p, of_tp(t, p), OFF, PAR, VN, LI, LJ, pw_x, pw_y)

v_closed(650.0, 30.0)
'''


def run(label, body, warm):
    '''Запускает пробу в свежем процессе; возвращает (мс, строки про кэш).'''
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / 'probe.py'
        f.write_text('import time\n_t0 = time.perf_counter()\n' + body
                     + '\nprint("ELAPSED", time.perf_counter() - _t0)\n')
        env = {'NUMBA_DEBUG_CACHE': '1', 'PATH': '/usr/bin:/bin',
               'HOME': str(Path.home())}
        runs, out = [], ''
        for i in range(4 if warm else 1):
            r = subprocess.run([sys.executable, str(f)], capture_output=True,
                               text=True, env={**dict(__import__('os').environ), **env})
            if r.returncode:
                print(f'  {label}: ОШИБКА\n{r.stderr.strip()[-800:]}')
                return None, ''
            out = r.stdout + r.stderr
            el = [l for l in r.stdout.splitlines() if l.startswith('ELAPSED')]
            if i:                     # первый прогон пишет кэш, его не считаем
                runs.append(float(el[0].split()[1]))
        return (min(runs) * 1e3 if runs else None), out


def cache_report(out):
    loaded = out.count('[cache] data loaded')
    saved = out.count('[cache] data saved')
    skipped = sum(1 for l in out.splitlines()
                  if 'cannot cache' in l.lower() or 'dynamic globals' in l.lower())
    return loaded, saved, skipped


print('=' * 72)
print('ОБЛАСТЬ 3: таблицы аргументами против замороженных в замыкании')
print('=' * 72)
for label, body in (('таблицы аргументами (как сейчас)', ARGS),
                    ('таблицы заморожены в замыкании', CLOSED)):
    ms, out = run(label, body, warm=True)
    if ms is None:
        continue
    loaded, saved, skipped = cache_report(out)
    print(f'\n  {label}')
    print(f'    холодный старт (горячий кэш):  {ms:7.1f} мс')
    print(f'    записей кэша загружено:        {loaded:7d}')
    print(f'    записей кэша сохранено:        {saved:7d}')
    print(f'    отказов кэшировать:            {skipped:7d}')
    for l in out.splitlines():
        if 'cannot cache' in l.lower() or 'dynamic globals' in l.lower():
            print(f'      {l.strip()[:100]}')
            break

print()
print('Если у замыкания отказов кэшировать нет и старт тот же — забираем его')
print('в пакет, скаляр области 3 удваивается бесплатно. Если кэш отваливается,')
print('решаем: 500 нс на вызов против лишних сотен миллисекунд на старте.')
