'''
Сравнение скорости if97_py с RustSEUIF97 (пакет seuif97).

Запуск:
    pip install seuif97
    python tests/bench_vs_rust.py

Единицы разные, и это главный источник бессмысленных сравнений:
    if97_py  принимает (T [K],   p [МПа])
    seuif97  принимает (p [МПа], t [°C])

Поэтому перед каждым замером значения сверяются между библиотеками. Если
расхождение выше RTOL, скрипт останавливается: сравнивать скорость разных
ответов смысла нет.

Замеряются два режима, потому что у библиотек разные сильные стороны:

  векторный  — if97_py получает numpy-массив и считает его одним вызовом
               ufunc'а; seuif97 скалярный, поэтому его приходится крутить
               питоновским циклом. Это режим «посчитать поле/сетку»;
  скалярный  — обе библиотеки вызываются по одной точке. Это режим
               «посчитать одно состояние в цикле расчёта схемы», и здесь
               видна стоимость одного вызова, а не пропускная способность.

Отдельно измеряется холодный старт: numba компилирует код при импорте, и
модуль if97/fluid/v.py объявлен без cache=True, поэтому регион 3
перекомпилируется в каждом новом процессе.
'''

import subprocess
import sys
import time

import numpy as np
from seuif97 import pt, pt2h, pt2s, pt2v

from if97 import bounds, fluid, steam, water

OW = 10  # id скорости звука в seuif97; прямой функции pt2w там нет

RTOL = 1e-6        # порог сверки значений
REPEATS = 5        # прогонов на замер, берём лучший
N_VEC = 200_000    # точек в векторном режиме
N_SCALAR = 50_000  # вызовов в скалярном режиме

# Боксы заведомо внутри одного региона. Принадлежность проверяется
# assert'ом ниже: если границы сдвинутся, скрипт упадёт, а не соврёт.
CASES = (
    ('регион 1 — вода',             1, (300.0, 620.0), (20.0, 50.0), ('v', 'h', 's', 'w')),
    ('регион 2 — перегретый пар',   2, (700.0, 1000.0), (0.5, 10.0), ('v', 'h', 's', 'w')),
    ('регион 3 — сверхкрит. флюид', 3, (630.0, 670.0), (30.0, 45.0), ('v', 'h')),
)

MINE = {
    1: {'v': water.v.t_p, 'h': water.h.t_p, 's': water.s.t_p, 'w': water.w.t_p},
    2: {'v': steam.v.t_p, 'h': steam.h.t_p, 's': steam.s.t_p, 'w': steam.w.t_p},
    # в регионе 3 h считается через плотность, s и w не реализованы
    3: {'v': fluid.v.t_p,
        'h': lambda t, p: fluid.h.t_ρ(t, 1.0 / fluid.v.t_p(t, p))},
}

THEIRS = {'v': pt2v, 'h': pt2h, 's': pt2s}


def sample(t_range, p_range, n, region):
    '''Детерминированная выборка точек внутри бокса, с проверкой региона.'''
    rng = np.random.default_rng(0)
    t = rng.uniform(t_range[0], t_range[1], n)
    p = rng.uniform(p_range[0], p_range[1], n)
    got = bounds.region_t_p(t, p)
    assert np.all(got == region), (
        f'бокс T={t_range} p={p_range} не лежит целиком в регионе {region}: '
        f'встретились {sorted(set(got.tolist()))}'
    )
    return t, p


def loop_theirs(name, p_list, tc_list):
    if name == 'w':
        for p, t_c in zip(p_list, tc_list):
            pt(p, t_c, OW)
    else:
        f = THEIRS[name]
        for p, t_c in zip(p_list, tc_list):
            f(p, t_c)


def repeat_theirs(name, p, t_c, n):
    if name == 'w':
        for _ in range(n):
            pt(p, t_c, OW)
    else:
        f = THEIRS[name]
        for _ in range(n):
            f(p, t_c)


def best(fn):
    out = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out)


def verify(region, name, t, p):
    '''Сверяет значения и возвращает максимальное отн. расхождение.'''
    t_c = t - 273.15
    mine = np.asarray(MINE[region][name](t, p), dtype=float)
    if name == 'w':
        theirs = np.array([pt(float(pi), float(ti), OW) for pi, ti in zip(p, t_c)])
    else:
        f = THEIRS[name]
        theirs = np.array([f(float(pi), float(ti)) for pi, ti in zip(p, t_c)])
    rel = np.abs(mine - theirs) / np.abs(theirs)
    i = int(rel.argmax())
    assert rel[i] <= RTOL, (
        f'{name}: расхождение {rel[i]:.3e} > {RTOL:.0e} при '
        f'T={t[i]:.3f} K, p={p[i]:.4f} МПа — if97_py {mine[i]!r} vs seuif97 {theirs[i]!r}'
    )
    return rel[i]


def cold_start():
    '''Время импорта + первого вызова в свежем процессе, в мс.'''
    probes = (
        ('if97, регионы 1 и 2',
         'from if97 import water, steam; water.h.t_p(300., 3.); steam.h.t_p(700., 1.)'),
        ('if97, регион 3',
         'from if97 import fluid; fluid.v.t_p(650., 30.)'),
        ('seuif97',
         'from seuif97 import pt2h; pt2h(3., 26.85)'),
    )
    print('\nХолодный старт (импорт + первый вызов, новый процесс):')
    for label, body in probes:
        code = f'import time; t0 = time.perf_counter(); {body}; print(time.perf_counter() - t0)'
        runs = []
        for _ in range(3):
            out = subprocess.run([sys.executable, '-c', code],
                                 capture_output=True, text=True, check=True)
            runs.append(float(out.stdout.strip()))
        print(f'  {label:24s} {min(runs) * 1e3:8.1f} мс')


def main():
    print('=' * 78)
    print('if97_py  vs  RustSEUIF97 (seuif97)')
    print('=' * 78)
    print(f'сверка: rtol <= {RTOL:.0e} | векторный режим: {N_VEC:,} точек | '
          f'скалярный: {N_SCALAR:,} вызовов | лучшее из {REPEATS}')

    for title, region, t_range, p_range, props in CASES:
        t_vec, p_vec = sample(t_range, p_range, N_VEC, region)
        t_chk, p_chk = sample(t_range, p_range, 2_000, region)
        p_list = p_vec.tolist()
        tc_list = (t_vec - 273.15).tolist()
        t0, p0 = float(t_vec[0]), float(p_vec[0])
        tc0 = t0 - 273.15

        print(f'\n{title}   T {t_range[0]:.0f}–{t_range[1]:.0f} K, '
              f'p {p_range[0]:g}–{p_range[1]:g} МПа')
        print(f'  {"":4s} {"сверка":>9s}   {"вектор, точек/с":>30s}   {"скаляр, точек/с":>30s}')
        print(f'  {"":4s} {"":>9s}   {"if97_py":>12s} {"seuif97":>10s} {"":>6s}   '
              f'{"if97_py":>12s} {"seuif97":>10s} {"":>6s}')

        for name in props:
            rel = verify(region, name, t_chk, p_chk)
            mine = MINE[region][name]

            mine(t_vec, p_vec)  # прогрев: numba компилирует при первом вызове
            v_mine = N_VEC / best(lambda: mine(t_vec, p_vec))
            v_theirs = N_VEC / best(lambda: loop_theirs(name, p_list, tc_list))

            s_mine = N_SCALAR / best(lambda: [mine(t0, p0) for _ in range(N_SCALAR)])
            s_theirs = N_SCALAR / best(lambda: repeat_theirs(name, p0, tc0, N_SCALAR))

            print(f'  {name:4s} {rel:9.1e}   {v_mine:12,.0f} {v_theirs:10,.0f} '
                  f'{v_mine / v_theirs:5.1f}x   '
                  f'{s_mine:12,.0f} {s_theirs:10,.0f} {s_mine / s_theirs:5.1f}x')

        if region == 3:
            print('  s, w в регионе 3 не реализованы в if97_py — пропущены')

    cold_start()

    print('\nЗамечания к интерпретации:')
    print('  - векторный режим льстит if97_py: seuif97 не имеет массивного API,')
    print('    и питоновский цикл вокруг него — часть замеряемого времени;')
    print('  - скалярный режим ближе к расчёту схемы по одному состоянию и')
    print('    показывает стоимость одного вызова;')
    print('  - h в регионе 3 у if97_py требует двух проходов (v, затем h(T, rho)),')
    print('    у seuif97 это один вызов pt2h;')
    print('  - seuif97 считает из (p, t); перевод K -> °C вынесен из замера.')


if __name__ == '__main__':
    main()
