'''
Безразмерная энергия Гиббса области 5 и её производные,
уравнения (33) и (34).

Идеально-газовая часть γ⁰ = ln π + Σ n⁰ τ^J⁰ даёт γ⁰_π = 1/π и γ⁰_ππ = -1/π²
аналитически, поэтому по таблице идут только производные по τ. Остаточная
часть γ^r обходится один раз на все пять своих производных.
'''

from math import log

from numba import njit

from .._jit import JIT
from . import tables
from .tables import NT, NT0, PSTAR, TSTAR

#: Таблицы для заморозки в тонких обёртках.
TAB = (tables.I, tables.J, tables.N, tables.J0, tables.N0)

@njit(**JIT)
def reduce(t, p):
    '''Приведённые переменные: π = p/1 МПа, τ = 1000 K/T.'''
    return p / PSTAR, TSTAR / t


@njit(**JIT)
def derivs(pi, tau, I, J, N, J0, N0):
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

    vj = tau
    gr = 0.0
    si = 0.0
    sii = 0.0
    sij = 0.0
    sj = 0.0
    sjj = 0.0
    for k in range(NT):
        ik = I[k]
        jk = J[k]
        item = N[k] * pi ** ik * vj ** jk
        i_item = ik * item
        j_item = jk * item
        gr += item
        si += i_item
        sii += (ik - 1) * i_item
        sij += jk * i_item
        sj += j_item
        sjj += (jk - 1) * j_item
    return (g0, s0t / tau, s0tt / (tau * tau), gr,
            si / pi, sii / (pi * pi), sj / vj, sjj / (vj * vj), sij / (pi * vj))
