'''Скорость звука (перегретый пар, область 2) по температуре и давлению, [м/с].

Соотношение стандарта: w²/(RT) = (1 + 2π γ^r_π + π² γ^r_π²) / [(1 - π² γ^r_ππ) + …]
'''

from math import sqrt

from numba import njit

from .._dispatch import make2
from .._jit import JIT
from ..consts import KJ, R
from .gibbs import TAB, derivs, reduce, scratch

I, J, N, J0, N0 = TAB


@njit(**JIT)
def kernel_core(t, p, pw_i, pw_j, I, J, N, J0, N0):
    pi, tau = reduce(t, p)
    (g0, g0_tau, g0_tautau, gr,
     gr_pi, gr_pipi, gr_tau, gr_tautau, gr_pitau) = derivs(pi, tau, pw_i, pw_j, I, J, N, J0, N0)
    g_tau = g0_tau + gr_tau
    g_tautau = g0_tautau + gr_tautau
    num = (1.0 + pi * gr_pi - tau * pi * gr_pitau) ** 2
    den = 1.0 - pi * pi * gr_pipi
    return sqrt(KJ * R * t * (1.0 + 2.0 * pi * gr_pi + pi * pi * gr_pi * gr_pi)
                / (den + num / (tau * tau * g_tautau)))


@njit(**JIT)
def kernel(t, p):
    pw_i, pw_j = scratch()
    return kernel_core(t, p, pw_i, pw_j, I, J, N, J0, N0)


@njit(**JIT)
def batch(t, p, out):
    pw_i, pw_j = scratch()          # один раз на весь массив, не на точку
    for k in range(t.shape[0]):
        out[k] = kernel_core(t[k], p[k], pw_i, pw_j, I, J, N, J0, N0)


#: Скаляр или массив; обе ветви зовут одно и то же ядро.
t_p = make2(kernel, batch)
