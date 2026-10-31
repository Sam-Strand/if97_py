'''
Полное состояние (вода, область 1) по температуре и давлению.

Возвращает namedtuple (v, rho, h, s, u, cp, cv, w). Стоит столько же, сколько
одно свойство: все восемь снимаются с одного обхода таблицы коэффициентов.
'''

from math import sqrt

from numba import njit

from .._dispatch import make_state
from .._jit import JIT
from ..consts import KJ, R
from .gibbs import TAB, derivs, reduce, scratch

I, J, N = TAB


@njit(**JIT)
def kernel_core(t, p, pw_i, pw_j, I, J, N):
    pi, tau = reduce(t, p)
    g, g_pi, g_pipi, g_tau, g_tautau, g_pitau = derivs(pi, tau, pw_i, pw_j, I, J, N)
    rt = R * t
    pi_g_pi = pi * g_pi
    v = rt * pi_g_pi / p / KJ
    dif = (g_pi - tau * g_pitau) ** 2
    return (v, 1.0 / v,
            rt * tau * g_tau,
            R * (tau * g_tau - g),
            rt * (tau * g_tau - pi_g_pi),
            -R * tau * tau * g_tautau,
            R * (-tau * tau * g_tautau + dif / g_pipi),
            sqrt(KJ * rt * g_pi * g_pi
                 / (dif / (tau * tau * g_tautau) - g_pipi)))


@njit(**JIT)
def kernel(t, p):
    pw_i, pw_j = scratch()
    return kernel_core(t, p, pw_i, pw_j, I, J, N)


@njit(**JIT)
def batch(t, p, out):
    pw_i, pw_j = scratch()          # один раз на весь массив, не на точку
    for k in range(t.shape[0]):
        (out[0, k], out[1, k], out[2, k], out[3, k],
         out[4, k], out[5, k], out[6, k], out[7, k]) = kernel_core(
            t[k], p[k], pw_i, pw_j, I, J, N)


t_p = make_state(kernel, batch)
