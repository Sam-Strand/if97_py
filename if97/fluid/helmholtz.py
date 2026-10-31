'''
Безразмерная энергия Гельмгольца области 3 и её производные, уравнение (28).

В области 3 независимые переменные — температура и плотность: вблизи
критической точки v(p,T) ведёт себя плохо, и пара (p,T) перестаёт быть
удобной. Все шесть производных снимаются за один обход таблицы (Таблица 32).

Лестницы степеней вместо возведения в степень
---------------------------------------------
Наивный обход считает `основание ** показатель` на каждый член таблицы, и
numba разворачивает это в бинарное возведение — шесть-одиннадцать умножений
на член. Вместо этого степени обоих оснований строятся один раз на точку
инкрементально (37 умножений), а внутренний цикл превращается в две
загрузки из массива и умножение. На 39 членах это даёт примерно 2.6x.

Буферы под лестницы приходят аргументами: в пакетном режиме они выделяются
один раз на весь массив, а не на каждую точку. `scratch()` их создаёт.
'''

from math import log

import numpy as np
from numba import njit

from .._jit import JIT
from . import tables
from .tables import N1, NT, RHOSTAR, TSTAR

IPOW = 12      # степени δ: 0..11
JPOW = 27      # степени τ: 0..26

#: Таблицы для заморозки в тонких обёртках.
TAB = (tables.I, tables.J, tables.N)

@njit(**JIT)
def scratch():
    '''Буферы под лестницы степеней.'''
    return np.empty(IPOW), np.empty(JPOW)


@njit(**JIT)
def reduce(t, rho):
    '''Приведённые переменные: τ = 647.096 K/T, δ = ρ/322 кг·м⁻³.'''
    return TSTAR / t, rho / RHOSTAR


@njit(**JIT)
def derivs(tau, delta, pw_i, pw_j, I, J, N):
    '''φ, φ_δ, φ_δδ, φ_τ, φ_ττ, φ_δτ.'''
    vi = delta
    vj = tau
    pw_i[0] = 1.0
    for m in range(1, 12):
        pw_i[m] = pw_i[m - 1] * vi
    pw_j[0] = 1.0
    for m in range(1, 27):
        pw_j[m] = pw_j[m - 1] * vj

    f = N1 * log(delta)
    sd = 0.0
    sdd = 0.0
    sdt = 0.0
    st = 0.0
    stt = 0.0
    for k in range(NT):
        ik = I[k]
        jk = J[k]
        item = N[k] * pw_i[ik] * pw_j[jk]
        i_item = ik * item
        j_item = jk * item
        f += item
        sd += i_item
        sdd += (ik - 1) * i_item
        sdt += jk * i_item
        st += j_item
        stt += (jk - 1) * j_item
    # член N1·ln δ даёт N1/δ и -N1/δ² в первой и второй производных по δ
    return (f, (N1 + sd) / delta, (-N1 + sdd) / (delta * delta),
            st / tau, stt / (tau * tau), sdt / (delta * tau))
