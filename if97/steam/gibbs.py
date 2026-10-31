'''
Безразмерная энергия Гиббса области 2 и её производные,
уравнения (16) и (17).

Идеально-газовая часть γ⁰ = ln π + Σ n⁰ τ^J⁰ даёт γ⁰_π = 1/π и γ⁰_ππ = -1/π²
аналитически, поэтому по её таблице идут только производные по τ; девять
членов не окупают лестницу, там обычное возведение в степень. Остаточная
часть γ^r обходится один раз на все пять своих производных.

Лестницы степеней вместо возведения в степень
---------------------------------------------
Наивный обход считает `основание ** показатель` на каждый член таблицы, и
numba разворачивает это в бинарное возведение — шесть-одиннадцать умножений
на член. Вместо этого степени обоих оснований строятся один раз на точку
инкрементально (82 умножений), а внутренний цикл превращается в две
загрузки из массива и умножение. На 43 членах это даёт примерно 2.6x.

Буферы под лестницы приходят аргументами: в пакетном режиме они выделяются
один раз на весь массив, а не на каждую точку. `scratch()` их создаёт.
'''

from math import log

import numpy as np
from numba import njit

from .._jit import JIT
from . import tables
from .tables import NT, NT0, PSTAR, TSTAR

IPOW = 25      # степени π: 0..24
JPOW = 59      # степени (τ-0.5): 0..58

#: Таблицы для заморозки в тонких обёртках.
TAB = (tables.I, tables.J, tables.N, tables.J0, tables.N0)

@njit(**JIT)
def scratch():
    '''Буферы под лестницы степеней остаточной части.'''
    return np.empty(IPOW), np.empty(JPOW)


@njit(**JIT)
def reduce(t, p):
    '''Приведённые переменные: π = p/1 МПа, τ = 540 K/T.'''
    return p / PSTAR, TSTAR / t


@njit(**JIT)
def derivs(pi, tau, pw_i, pw_j, I, J, N, J0, N0):
    '''γ⁰, γ⁰_τ, γ⁰_ττ, γ^r, γ^r_π, γ^r_ππ, γ^r_τ, γ^r_ττ, γ^r_πτ.'''
    g0 = log(pi)
    s0t = 0.0
    s0tt = 0.0
    for k in range(NT0):
        jk = J0[k]
        item = N0[k] * tau ** jk
        g0 += item
        s0t += jk * item
        s0tt += jk * (jk - 1) * item

    vi = pi
    vj = tau - 0.5
    pw_i[0] = 1.0
    for m in range(1, 25):
        pw_i[m] = pw_i[m - 1] * vi
    pw_j[0] = 1.0
    for m in range(1, 59):
        pw_j[m] = pw_j[m - 1] * vj

    gr = 0.0
    si = 0.0
    sii = 0.0
    sij = 0.0
    sj = 0.0
    sjj = 0.0
    for k in range(NT):
        ik = I[k]
        jk = J[k]
        item = N[k] * pw_i[ik] * pw_j[jk]
        i_item = ik * item
        j_item = jk * item
        gr += item
        si += i_item
        sii += (ik - 1) * i_item
        sij += jk * i_item
        sj += j_item
        sjj += (jk - 1) * j_item
    return (g0, s0t / tau, s0tt / (tau * tau), gr,
            si / vi, sii / (vi * vi), sj / vj, sjj / (vj * vj), sij / (vi * vj))
