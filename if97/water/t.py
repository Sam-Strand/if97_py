'''
Температура воды (область 1) по давлению с энтальпией или энтропией, [K].

Обратные уравнения (11) и (13) стандарта, Таблицы 6 и 8. В отличие от
прямого уравнения (7) это подгонки, а не обращения: стандарт не требует
от них точности и допускает расхождение с прямым уравнением до 25 мК.

Соотношения стандарта: θ = Σ nᵢ π^Iᵢ (η + 1)^Jᵢ,  η = h/h*
                       θ = Σ nᵢ π^Iᵢ (σ + 2)^Jᵢ,  σ = s/s*
'''

import numpy as np
from numba import njit

from .._dispatch import make2
from .._jit import JIT
from .._poly import summed
from . import tables_backward as tb
from .tables_backward import (HSTAR, PH_XHI, PH_XLO, PH_YHI, PH_YLO, PS_XHI,
                              PS_XLO, PS_YHI, PS_YLO, SSTAR, XPOW, YPOW)

PH = (tb.PH_I, tb.PH_J, tb.PH_N)
PS = (tb.PS_I, tb.PS_J, tb.PS_N)
PH_I, PH_J, PH_N = PH
PS_I, PS_J, PS_N = PS


@njit(**JIT)
def scratch():
    return np.empty(XPOW), np.empty(YPOW)


@njit(**JIT)
def p_h_core(p, h, pw_x, pw_y, I, J, N):
    return summed(p, h / HSTAR + 1.0, I, J, N,
                  PH_XLO, PH_XHI, PH_YLO, PH_YHI, pw_x, pw_y)


@njit(**JIT)
def p_h_kernel(p, h):
    pw_x, pw_y = scratch()
    return p_h_core(p, h, pw_x, pw_y, PH_I, PH_J, PH_N)


@njit(**JIT)
def p_h_batch(p, h, out):
    pw_x, pw_y = scratch()          # один раз на весь массив, не на точку
    for k in range(p.shape[0]):
        out[k] = p_h_core(p[k], h[k], pw_x, pw_y, PH_I, PH_J, PH_N)


@njit(**JIT)
def p_s_core(p, s, pw_x, pw_y, I, J, N):
    return summed(p, s / SSTAR + 2.0, I, J, N,
                  PS_XLO, PS_XHI, PS_YLO, PS_YHI, pw_x, pw_y)


@njit(**JIT)
def p_s_kernel(p, s):
    pw_x, pw_y = scratch()
    return p_s_core(p, s, pw_x, pw_y, PS_I, PS_J, PS_N)


@njit(**JIT)
def p_s_batch(p, s, out):
    pw_x, pw_y = scratch()
    for k in range(p.shape[0]):
        out[k] = p_s_core(p[k], s[k], pw_x, pw_y, PS_I, PS_J, PS_N)


#: Скаляр или массив; обе ветви зовут одно и то же ядро.
p_h = make2(p_h_kernel, p_h_batch)
p_s = make2(p_s_kernel, p_s_batch)
