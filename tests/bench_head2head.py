'''
Очная ставка: if97 против RustSEUIF97 (пакет seuif97).

Запуск:
    pip install seuif97
    python tests/bench_head2head.py
    python tests/bench_head2head.py --quick    # малые выборки, только на вменяемость

Единицы: if97 (T[K], p[МПа]), seuif97 (p[МПа], t[°C]). Значения сверяются
до замера; при расхождении выше допуска скрипт падает.

Замеряются две разные вещи, и путать их не стоит. «Вектор» — один вызов на
весь массив: у if97 граница Python пересекается один раз, у seuif97 — на
каждую точку, потому что обёртка скалярная. «Скаляр» — вызов на точку у
обоих. Первое показывает пропускную способность, второе — накладные расходы
на вызов, и выигрывают они в разном.
'''

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from seuif97 import ph2t, ps2t, pt, pt2h, pt2s, pt2v, px2t, tx2p

from if97 import bounds, if97

OW = 10
RTOL = 1e-9
#: Обратным уравнениям стандарт сам отводит 25 мК, поэтому строгого
#: совпадения с чужой реализацией от них ждать нельзя: она вправе уточнять
#: результат итерацией по прямому уравнению.
RTOL_BACK = 1e-4

QUICK = '--quick' in sys.argv
REPEATS = 2 if QUICK else 5
N_VEC = 2_000 if QUICK else 200_000
N_SCALAR = 500 if QUICK else 50_000
NC = 200 if QUICK else 2000


def best(fn):
    out = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out)


def sca(fn):
    def go():
        for _ in range(N_SCALAR):
            fn()
    return N_SCALAR / best(go)


def sample(region, t_lo, t_hi, p_lo, p_hi, n=N_VEC):
    '''n точек внутри области: отбор с отбраковкой, границы не срезаются.'''
    rng = np.random.default_rng(0)
    ts, ps, got = [], [], 0
    while got < n:
        t = rng.uniform(t_lo, t_hi, n)
        p = rng.uniform(p_lo, p_hi, n)
        m = bounds.region_t_p(t, p) == region
        ts.append(t[m]); ps.append(p[m]); got += int(m.sum())
    t, p = np.concatenate(ts)[:n], np.concatenate(ps)[:n]
    return t, p


def check(label, got, ref, tol=RTOL, kelvin=False):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    rel = np.abs(got / ref - 1)
    i = int(rel.argmax())
    extra = f' ({abs(got[i] - ref[i]) * 1e3:.1f} мК)' if kelvin else ''
    assert rel[i] <= tol, f'{label}: расхождение {rel[i]:.2e}{extra}'
    print(f'    сверка {label:26s} max rel = {rel[i]:.2e}{extra}')


def section(title, note=None):
    print('=' * 70)
    print(title)
    print('=' * 70)
    if note:
        print(f'    {note}')


def duel(mine_vec, mine_sca, their_vec, their_sca, n=N_VEC, note=None):
    '''Прогрев, замер, итог. Возвращает отношения вектор/скаляр.'''
    mine_vec()
    print(f'\n    {"":9s} {"вектор, точек/с":>18s} {"скаляр, точек/с":>18s}')
    a = (n / best(mine_vec), sca(mine_sca))
    b = (n / best(their_vec), sca(their_sca))
    for label, (vec, scl) in (('if97', a), ('seuif97', b)):
        print(f'    {label:9s} {vec:18,.0f} {scl:18,.0f}')
    print(f'    {"итог":9s} вектор {a[0] / b[0]:.2f}x   скаляр {a[1] / b[1]:.2f}x')
    if note:
        print(f'    {note}')
    print()


# ======================================================================
# ПРЯМЫЕ УРАВНЕНИЯ
# ======================================================================
print()
print('#' * 70)
print('# ПРЯМЫЕ УРАВНЕНИЯ: свойства по T и p')
print('#' * 70)
print()

FORWARD = (
    ('ОБЛАСТЬ 1 (вода)', 1, 'water', (300., 620., 20., 50.)),
    ('ОБЛАСТЬ 2 (перегретый пар)', 2, 'steam', (700., 1070., 1., 10.)),
    ('ОБЛАСТЬ 5 (высокотемпературный пар)', 5, 'gas', (1100., 2200., 1., 50.)),
)

for title, reg, area, box in FORWARD:
    mod = getattr(if97, area)
    t, p = sample(reg, *box)
    tc = t - 273.15
    pl, tcl = p.tolist(), tc.tolist()
    t0, p0, tc0 = float(t[0]), float(p[0]), float(tc[0])

    section(f'{title}, скорость звука w     '
            f'T {box[0]:.0f}-{box[1]:.0f} K, p {box[2]:g}-{box[3]:g} МПа')
    check('seuif97 vs if97, w', [pt(a, b, OW) for a, b in zip(pl[:NC], tcl[:NC])],
          mod.w.t_p(t[:NC], p[:NC]))
    duel(lambda: mod.w.t_p(t, p), lambda: mod.w.t_p(t0, p0),
         lambda: [pt(x, y, OW) for x, y in zip(pl, tcl)],
         lambda: pt(p0, tc0, OW))

    section(f'{title}, состояние: v + rho + h + s + u + cp + cv + w')

    def theirs_all_vec(pl=pl, tcl=tcl):
        for x, y in zip(pl, tcl):
            pt2v(x, y); pt2h(x, y); pt2s(x, y); pt(x, y, OW)

    duel(lambda: mod.state.t_p(t, p), lambda: mod.state.t_p(t0, p0),
         theirs_all_vec,
         lambda: (pt2v(p0, tc0), pt2h(p0, tc0), pt2s(p0, tc0), pt(p0, tc0, OW)),
         note='(if97 считает восемь свойств, seuif97 — четыре четырьмя вызовами FFI)')

# ---------------------------- область 3 ----------------------------
t, p = sample(3, 630., 670., 30., 45.)
tc = t - 273.15
pl, tcl = p.tolist(), tc.tolist()
t0, p0, tc0 = float(t[0]), float(p[0]), float(tc[0])

section('ОБЛАСТЬ 3 (сверхкритический флюид), удельный объём v     '
        'T 630-670 K, p 30-45 МПа',
        note='у if97 это обратное уравнение SR5-05 v(p,T), 26 подобластей')
check('seuif97 vs if97, v', [pt2v(a, b) for a, b in zip(pl[:NC], tcl[:NC])],
      if97.fluid.v.t_p(t[:NC], p[:NC]))
duel(lambda: if97.fluid.v.t_p(t, p), lambda: if97.fluid.v.t_p(t0, p0),
     lambda: [pt2v(x, y) for x, y in zip(pl, tcl)], lambda: pt2v(p0, tc0))

section('ОБЛАСТЬ 3, состояние: v + rho + h + s + u + cp + cv + w')
duel(lambda: if97.fluid.state.t_p(t, p), lambda: if97.fluid.state.t_p(t0, p0),
     lambda: [(pt2v(x, y), pt2h(x, y), pt2s(x, y), pt(x, y, OW))
              for x, y in zip(pl, tcl)],
     lambda: (pt2v(p0, tc0), pt2h(p0, tc0), pt2s(p0, tc0), pt(p0, tc0, OW)),
     note='(if97 считает восемь свойств, seuif97 — четыре четырьмя вызовами FFI)')

# ---------------------------- линия насыщения ----------------------------
rng = np.random.default_rng(0)
ts = rng.uniform(280., 640., N_VEC)
tsl = (ts - 273.15).tolist()
ts0, tsc0 = float(ts[0]), float(tsl[0])

section('ЛИНИЯ НАСЫЩЕНИЯ, p_s(T)     T 280-640 K, ур. (30)')
check('seuif97 vs if97, p_s(T)', [tx2p(y, 0.) for y in tsl[:NC]],
      bounds.saturationPressure_t(ts[:NC]))
duel(lambda: bounds.saturationPressure_t(ts),
     lambda: bounds.saturationPressure_t(ts0),
     lambda: [tx2p(y, 0.) for y in tsl], lambda: tx2p(tsc0, 0.))

ps_ = rng.uniform(0.001, 22., N_VEC)
psl = ps_.tolist()
ps0 = float(ps_[0])

section('ЛИНИЯ НАСЫЩЕНИЯ, T_s(p)     p 0.001-22 МПа, ур. (31)')
check('seuif97 vs if97, T_s(p)', [px2t(x, 0.) + 273.15 for x in psl[:NC]],
      bounds.saturationTemp_p(ps_[:NC]))
duel(lambda: bounds.saturationTemp_p(ps_),
     lambda: bounds.saturationTemp_p(ps0),
     lambda: [px2t(x, 0.) for x in psl], lambda: px2t(ps0, 0.))

# ======================================================================
# ОБРАТНЫЕ УРАВНЕНИЯ
# ======================================================================
print('#' * 70)
print('# ОБРАТНЫЕ УРАВНЕНИЯ: T по p с энтальпией или энтропией')
print('#' * 70)
print()
print('Асимметрия, которую надо держать в голове: ph2t/ps2t у seuif97 сами')
print('определяют область по (p,h), а water.t.p_h и steam.t.p_h — нет, они')
print('работают как и прямые уравнения по областям. То есть здесь if97 делает')
print('меньше работы, и чистым сравнением уравнение-против-уравнения эти')
print('числа не являются.')
print()

BACKWARD = (
    ('ОБЛАСТЬ 1', 1, 'water', (300., 620., 20., 100.), None),
    ('ОБЛАСТЬ 2', 2, 'steam', (280., 1070., 0.001, 100.), 'вся область, p равномерно'),
    ('ОБЛАСТЬ 2, только 2a', 2, 'steam', (300., 1070., 0.001, 4.), None),
    ('ОБЛАСТЬ 2, 2b и 2c', 2, 'steam', (700., 1070., 10., 100.), None),
)


def split_ph(p, h):
    '''Разбивка по подобластям так, как её делает сам диспетчер T(p,h).'''
    a = p <= if97.steam.t.P_2AB
    above = p >= bounds.p2bcMin
    hb = bounds.b2bcEnthalpy_p(np.where(above, p, bounds.p2bcMin))
    c = ~a & above & (h < hb)
    return (f'подобласти: 2a {int(a.sum())}, 2b {int((~a & ~c).sum())}, '
            f'2c {int(c.sum())}')


def split_ps(p, s):
    '''Для T(p,s) граница 2b/2c другая: не линия B2bc, а s = 5.85.'''
    a = p <= if97.steam.t.P_2AB
    c = ~a & (s < if97.steam.t.S_2BC)
    return (f'подобласти: 2a {int(a.sum())}, 2b {int((~a & ~c).sum())}, '
            f'2c {int(c.sum())}')


for title, reg, area, box, tag in BACKWARD:
    mod = getattr(if97, area)
    t, p = sample(reg, *box)
    h = np.asarray(mod.h.t_p(t, p), float)
    s = np.asarray(mod.s.t_p(t, p), float)
    pl, hl, sl = p.tolist(), h.tolist(), s.tolist()
    p0, h0, s0 = float(p[0]), float(h[0]), float(s[0])
    tail = (f'{tag}; ' if tag else '') + f'p {box[2]:g}-{box[3]:g} МПа'

    section(f'{title}, T(p,h)     {tail}',
            note=split_ph(p, h) if reg == 2 else None)
    check('seuif97 vs if97, T(p,h)',
          [ph2t(x, y) + 273.15 for x, y in zip(pl[:NC], hl[:NC])],
          mod.t.p_h(p[:NC], h[:NC]), RTOL_BACK, kelvin=True)
    duel(lambda: mod.t.p_h(p, h), lambda: mod.t.p_h(p0, h0),
         lambda: [ph2t(x, y) for x, y in zip(pl, hl)], lambda: ph2t(p0, h0))

    section(f'{title}, T(p,s)     {tail}',
            note=split_ps(p, s) if reg == 2 else None)
    check('seuif97 vs if97, T(p,s)',
          [ps2t(x, y) + 273.15 for x, y in zip(pl[:NC], sl[:NC])],
          mod.t.p_s(p[:NC], s[:NC]), RTOL_BACK, kelvin=True)
    duel(lambda: mod.t.p_s(p, s), lambda: mod.t.p_s(p0, s0),
         lambda: [ps2t(x, y) for x, y in zip(pl, sl)], lambda: ps2t(p0, s0))

# ---------------------------- граница B2bc ----------------------------
section('ГРАНИЦА B2bc, h(p)     p 6.5467-100 МПа',
        note='(у seuif97 эта граница не вынесена в питоновский API, сравнивать не с чем)')
pb = np.random.default_rng(0).uniform(bounds.p2bcMin, 100., N_VEC)
pb0 = float(pb[0])
bounds.b2bcEnthalpy_p(pb)
print(f'\n    {"":9s} {"вектор, точек/с":>18s} {"скаляр, точек/с":>18s}')
print(f'    {"if97":9s} {N_VEC / best(lambda: bounds.b2bcEnthalpy_p(pb)):18,.0f} '
      f'{sca(lambda: bounds.b2bcEnthalpy_p(pb0)):18,.0f}')
print()

# ======================================================================
# ХОЛОДНЫЙ СТАРТ
# ======================================================================
print('#' * 70)
print('# ХОЛОДНЫЙ СТАРТ: свежий процесс, лучшее из 3')
print('#' * 70)
print()
print('Импорт и первый вызов разделены, чтобы видеть, где именно уходит время.')
print('У seuif97 все пробы дают одно число: загрузка эагерная, всё собрано')
print('заранее. У if97 numba компилирует по требованию, поэтому важно, сколько')
print('областей тронуто — последние строки трогают реалистичный набор.')
print()

PROBES = (
    ('import numpy', 'import numpy'),
    ('import numba', 'import numba'),
    ('', ''),
    ('seuif97: импорт', 'import seuif97'),
    ('seuif97: + первый вызов',
     'import seuif97\nassert abs(seuif97.pt2h(3., 26.85) - 115.33) < 0.1'),
    ('seuif97: + восемь свойств',
     'import seuif97 as s\n'
     'for f in (s.pt2h, s.pt2s, s.pt2v):\n    f(3., 26.85)\n'
     'for o in (2, 7, 8, 9, 10):\n    s.pt(3., 26.85, o)'),
    ('seuif97: + обратные', 'import seuif97 as s\ns.ph2t(3., 500.)\ns.ps2t(3., 0.5)'),
    ('', ''),
    ('if97: импорт', 'import if97'),
    ('if97: + одно свойство (обл. 1)',
     'from if97 import if97\n'
     'assert abs(if97.water.h.t_p(300., 3.) - 115.33) < 0.1'),
    ('if97: + состояние (обл. 1)',
     'from if97 import if97\n'
     'assert abs(if97.water.state.t_p(300., 3.).h - 115.33) < 0.1'),
    ('if97: + обратные (обл. 1)',
     'from if97 import if97\n'
     'assert abs(if97.water.t.p_h(3., 500.) - 391.8) < 0.1\n'
     'if97.water.t.p_s(3., 0.5)'),
    ('if97: + обратные (обл. 2, три подобл.)',
     'from if97 import if97\n'
     'if97.steam.t.p_h(3., 3000.)\nif97.steam.t.p_h(60., 2700.)\n'
     'if97.steam.t.p_s(8., 6.)'),
    ('if97: + состояния 1, 2, 3, 5',
     'from if97 import if97\n'
     'if97.water.state.t_p(300., 3.)\n'
     'if97.steam.state.t_p(700., 1.)\n'
     'if97.fluid.state.t_p(650., 30.)\n'
     'if97.gas.state.t_p(1500., 30.)'),
    ('if97: + автовыбор области',
     'from if97 import if97\nif97.state_t_p(650., 30.)'),
    ('if97: + всё, включая обратные',
     'from if97 import if97\n'
     'if97.state_t_p(650., 30.)\n'
     'if97.water.t.p_h(3., 500.)\nif97.water.t.p_s(3., 0.5)\n'
     'if97.steam.t.p_h(3., 3000.)\nif97.steam.t.p_h(60., 2700.)\n'
     'if97.steam.t.p_s(8., 6.)'),
)

with tempfile.TemporaryDirectory() as td:
    for label, body in PROBES:
        if not label:
            print()
            continue
        f = Path(td) / f'probe_{abs(hash(label))}.py'
        f.write_text('import time\n_t0 = time.perf_counter()\n' + body
                     + '\nprint(time.perf_counter() - _t0)\n')
        runs = []
        for _ in range(3):
            r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
            if r.returncode:
                print(f'    {label:38s}  ОШИБКА: {r.stderr.strip().splitlines()[-1]}')
                break
            runs.append(float(r.stdout.strip()))
        else:
            print(f'    {label:38s} {min(runs) * 1e3:9.1f} мс')

print()
print('Было в v1.1.2: области 1-2 — 1729 мс, область 3 — 8498 мс.')
print('Импорт областей ленивый: платится только за ту, которую тронули.')
