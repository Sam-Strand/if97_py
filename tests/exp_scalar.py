'''
Из чего складывается скалярный вызов.

В векторном режиме пакет обгоняет seuif97, в скалярном отстаёт втрое. Этот
скрипт раскладывает стоимость одного вызова на составляющие, чтобы понять,
что из отставания наше, а что — пол numba.

Запуск:
    python tests/exp_scalar.py

Ничего в пакете не меняет, только замеряет варианты.
'''

import time

import numpy as np
from numba import njit

from if97 import if97
from if97._jit import JIT
from if97.fluid import subregion, tables
from if97.water import gibbs
from if97.water import w as water_w

N = 200_000
REPEATS = 7


def best(fn):
    out = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        out.append(time.perf_counter() - t0)
    return min(out) / N * 1e9      # нс на вызов


def show(label, ns, base=None):
    tail = f'   {ns - base:+8.0f} нс' if base is not None else ''
    print(f'  {label:52s} {ns:8.0f} нс{tail}')
    return ns


# --------------------------------------------------------------------------
# 1. Пол: сколько стоит пересечь границу Python -> numba вообще
# --------------------------------------------------------------------------

@njit(**JIT)
def nop2(a, b):
    return a + b


@njit('float64(float64, float64)', **JIT)
def nop2_sig(a, b):
    return a + b


@njit(**JIT)
def nop7(a, b, c, d, e, f, g):
    return a + b


print('=' * 72)
print('ПОЛ: пустое ядро, вызов из Python')
print('=' * 72)
nop2(1.0, 2.0)
nop2_sig(1.0, 2.0)
A = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)
nop7(1.0, 2.0, *A)
floor2 = show('njit(a, b), ленивая компиляция', best(lambda: nop2(1.0, 2.0)))
show('njit(a, b), явная сигнатура', best(lambda: nop2_sig(1.0, 2.0)), floor2)
show('njit(a, b, + 5 массивов)', best(lambda: nop7(1.0, 2.0, *A)), floor2)
show('чистый питоновский вызов f(a, b)',
     best(lambda: (lambda x, y: x)(1.0, 2.0)))

# --------------------------------------------------------------------------
# 2. Область 1: обёртка, ядро, публичная функция
# --------------------------------------------------------------------------
print()
print('=' * 72)
print('ОБЛАСТЬ 1, скорость звука: где уходит время')
print('=' * 72)
k = water_w.kernel
k(300.0, 30.0)
core = water_w.kernel_core
pw_i, pw_j = gibbs.scratch()
core(300.0, 30.0, pw_i, pw_j)

a = show('ядро kernel_core(t, p, буферы)', best(lambda: core(300.0, 30.0, pw_i, pw_j)))
b = show('ядро kernel(t, p) — само выделяет буферы', best(lambda: k(300.0, 30.0)), a)
c = show('публичная water.w.t_p(t, p)', best(lambda: if97.water.w.t_p(300.0, 30.0)), b)
print(f'  {"из них пол numba:":52s} {floor2:8.0f} нс'
      f'   ({floor2 / c * 100:.0f}% от публичного вызова)')

# --------------------------------------------------------------------------
# 3. Область 3: цена пяти массивов в сигнатуре
# --------------------------------------------------------------------------
print()
print('=' * 72)
print('ОБЛАСТЬ 3, удельный объём: цена массивов в сигнатуре')
print('=' * 72)

# вариант А: как сейчас — пять массивов аргументами
v_args = subregion.v_pt
v_args(650.0, 30.0, *A)

# вариант Б: таблицы заморожены в замыкании, сигнатура (float64, float64)
def _make_closed(off, par, vn, li, lj):
    v_reg = subregion.v_reg
    of_tp = subregion.of_tp
    scratch = subregion.scratch

    @njit(**JIT)
    def v_closed(t, p):
        pw_x, pw_y = scratch()
        return v_reg(t, p, of_tp(t, p), off, par, vn, li, lj, pw_x, pw_y)
    return v_closed


v_closed = _make_closed(*A)
v_closed(650.0, 30.0)

# вариант В: то же, но с явной сигнатурой
def _make_closed_sig(off, par, vn, li, lj):
    v_reg = subregion.v_reg
    of_tp = subregion.of_tp
    scratch = subregion.scratch

    @njit('float64(float64, float64)', **JIT)
    def v_closed_sig(t, p):
        pw_x, pw_y = scratch()
        return v_reg(t, p, of_tp(t, p), off, par, vn, li, lj, pw_x, pw_y)
    return v_closed_sig


v_closed_sig = _make_closed_sig(*A)
v_closed_sig(650.0, 30.0)

assert abs(v_args(650.0, 30.0, *A) / v_closed(650.0, 30.0) - 1) < 1e-14
assert abs(v_args(650.0, 30.0, *A) / v_closed_sig(650.0, 30.0) - 1) < 1e-14

x = show('ядро с 5 массивами в сигнатуре', best(lambda: v_args(650.0, 30.0, *A)))
show('то же, таблицы заморожены в замыкании', best(lambda: v_closed(650.0, 30.0)), x)
show('замыкание + явная сигнатура', best(lambda: v_closed_sig(650.0, 30.0)), x)
show('публичная fluid.v.t_p(t, p)', best(lambda: if97.fluid.v.t_p(650.0, 30.0)), x)

print()
print('Если замыкание заметно быстрее, таблицы стоит заморозить в ядрах. Цена')
print('вопроса — кэш: numba не кэширует функции с замороженными массивами.')
print('Проверить, пишется ли кэш:')
print('  NUMBA_DEBUG_CACHE=1 python -c "from if97 import if97;'
      ' if97.fluid.v.t_p(650.,30.)" 2>&1 | grep -c "data loaded"')
