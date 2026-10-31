'''
Температура пара (область 2) по давлению с энтальпией или энтропией, [K].

Одним полиномом область 2 не покрывается, поэтому для обратных уравнений
она разбита на три подобласти (Рис. 2 стандарта): 2a ниже 4 МПа, 2b и 2c
выше. Для T(p,h) их разделяет линия B2bc, для T(p,s) — просто значение
s = 5.85 кДж/(кг·K), на которое эта линия и настроена. Уравнения (22)-(27),
Таблицы 20-22 и 25-27; допуск стандарта — те же 25 мК, что и в области 1.

Соотношения стандарта:
    2a  θ = Σ nᵢ π^Iᵢ (η − 2.1)^Jᵢ        2a  θ = Σ nᵢ π^Iᵢ (σ − 2)^Jᵢ
    2b  θ = Σ nᵢ (π − 2)^Iᵢ (η − 2.6)^Jᵢ   2b  θ = Σ nᵢ π^Iᵢ (10 − σ)^Jᵢ
    2c  θ = Σ nᵢ (π + 25)^Iᵢ (η − 1.8)^Jᵢ  2c  θ = Σ nᵢ π^Iᵢ (2 − σ)^Jᵢ

Показатели Iᵢ уравнения (25) кратны 1/4, поэтому лестница там строится по
π^(1/4), а в таблице лежит 4Iᵢ — целое.
'''

import numpy as np
from numba import njit

from . import tables_backward as tb
from .. import b2bc
from .._dispatch import make2
from .._jit import JIT, JIT_NAN
from .._poly import summed
from .tables_backward import (AH_XHI, AH_XLO, AH_YHI, AH_YLO, AS_XHI, AS_XLO,
                              AS_YHI, AS_YLO, BH_XHI, BH_XLO, BH_YHI, BH_YLO,
                              BS_XHI, BS_XLO, BS_YHI, BS_YLO, CH_XHI, CH_XLO,
                              CH_YHI, CH_YLO, CS_XHI, CS_XLO, CS_YHI, CS_YLO,
                              HSTAR, SSTAR_A, SSTAR_B, SSTAR_C, XPOW, YPOW)

#: Граница подобласти 2a, [МПа]: ниже неё одно уравнение покрывает всю область.
P_2AB = 4.0
#: Граница подобластей 2b и 2c для T(p,s), [кДж/(кг·K)].
S_2BC = 5.85

AH_I, AH_J, AH_N = tb.AH_I, tb.AH_J, tb.AH_N
BH_I, BH_J, BH_N = tb.BH_I, tb.BH_J, tb.BH_N
CH_I, CH_J, CH_N = tb.CH_I, tb.CH_J, tb.CH_N
AS_I, AS_J, AS_N = tb.AS_I, tb.AS_J, tb.AS_N
BS_I, BS_J, BS_N = tb.BS_I, tb.BS_J, tb.BS_N
CS_I, CS_J, CS_N = tb.CS_I, tb.CS_J, tb.CS_N

PMIN = b2bc.PMIN
b2bc_h_p = b2bc.h_p_kernel


@njit(**JIT)
def scratch():
    return np.empty(XPOW), np.empty(YPOW)


# ------------------------------- T(p,h) -------------------------------

@njit(**JIT)
def ah_core(p, h, pw_x, pw_y, I, J, N):
    '''Подобласть 2a, уравнение (22).'''
    return summed(p, h / HSTAR - 2.1, I, J, N,
                  AH_XLO, AH_XHI, AH_YLO, AH_YHI, pw_x, pw_y)


@njit(**JIT)
def bh_core(p, h, pw_x, pw_y, I, J, N):
    '''Подобласть 2b, уравнение (23).'''
    return summed(p - 2.0, h / HSTAR - 2.6, I, J, N,
                  BH_XLO, BH_XHI, BH_YLO, BH_YHI, pw_x, pw_y)


@njit(**JIT)
def ch_core(p, h, pw_x, pw_y, I, J, N):
    '''Подобласть 2c, уравнение (24).'''
    return summed(p + 25.0, h / HSTAR - 1.8, I, J, N,
                  CH_XLO, CH_XHI, CH_YLO, CH_YHI, pw_x, pw_y)


@njit(**JIT_NAN)
def p_h_core(p, h, pw_x, pw_y):
    '''Выбор подобласти и расчёт. Таблицы заморожены: ветвей три, а не одна.

    fastmath выключен только здесь: тело — одни сравнения, ускорять нечего,
    зато ветвление остаётся определённым, если на вход пришёл NaN. Сами
    полиномы ниже компилируются с fastmath, как и в прямых уравнениях.
    '''
    if p <= P_2AB:
        return ah_core(p, h, pw_x, pw_y, AH_I, AH_J, AH_N)
    if p < PMIN:                    # линия B2bc начинается только с PMIN
        return bh_core(p, h, pw_x, pw_y, BH_I, BH_J, BH_N)
    if h >= b2bc_h_p(p):
        return bh_core(p, h, pw_x, pw_y, BH_I, BH_J, BH_N)
    return ch_core(p, h, pw_x, pw_y, CH_I, CH_J, CH_N)


@njit(**JIT_NAN)
def p_h_kernel(p, h):
    pw_x, pw_y = scratch()
    return p_h_core(p, h, pw_x, pw_y)


@njit(**JIT_NAN)
def p_h_batch(p, h, out):
    pw_x, pw_y = scratch()          # один раз на весь массив, не на точку
    for k in range(p.shape[0]):
        out[k] = p_h_core(p[k], h[k], pw_x, pw_y)


# ------------------------------- T(p,s) -------------------------------

@njit(**JIT)
def as_core(p, s, pw_x, pw_y, I, J, N):
    '''Подобласть 2a, уравнение (25): лестница по π^(1/4), показатели — 4Iᵢ.'''
    return summed(p ** 0.25, s / SSTAR_A - 2.0, I, J, N,
                  AS_XLO, AS_XHI, AS_YLO, AS_YHI, pw_x, pw_y)


@njit(**JIT)
def bs_core(p, s, pw_x, pw_y, I, J, N):
    '''Подобласть 2b, уравнение (26).'''
    return summed(p, 10.0 - s / SSTAR_B, I, J, N,
                  BS_XLO, BS_XHI, BS_YLO, BS_YHI, pw_x, pw_y)


@njit(**JIT)
def cs_core(p, s, pw_x, pw_y, I, J, N):
    '''Подобласть 2c, уравнение (27).'''
    return summed(p, 2.0 - s / SSTAR_C, I, J, N,
                  CS_XLO, CS_XHI, CS_YLO, CS_YHI, pw_x, pw_y)


@njit(**JIT_NAN)
def p_s_core(p, s, pw_x, pw_y):
    '''Выбор подобласти и расчёт: граница 2b/2c здесь просто s = 5.85.'''
    if p <= P_2AB:
        return as_core(p, s, pw_x, pw_y, AS_I, AS_J, AS_N)
    if s >= S_2BC:
        return bs_core(p, s, pw_x, pw_y, BS_I, BS_J, BS_N)
    return cs_core(p, s, pw_x, pw_y, CS_I, CS_J, CS_N)


@njit(**JIT_NAN)
def p_s_kernel(p, s):
    pw_x, pw_y = scratch()
    return p_s_core(p, s, pw_x, pw_y)


@njit(**JIT_NAN)
def p_s_batch(p, s, out):
    pw_x, pw_y = scratch()
    for k in range(p.shape[0]):
        out[k] = p_s_core(p[k], s[k], pw_x, pw_y)


#: Скаляр или массив; обе ветви зовут одно и то же ядро.
p_h = make2(p_h_kernel, p_h_batch)
p_s = make2(p_s_kernel, p_s_batch)
