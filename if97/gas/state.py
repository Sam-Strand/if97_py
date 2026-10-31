'''
Полное состояние (высокотемпературный пар, область 5) по температуре и давлению.

Возвращает namedtuple (v, rho, h, s, u, cp, cv, w). Стоит столько же, сколько
одно свойство: все восемь снимаются с одного обхода таблицы коэффициентов.
'''

from math import sqrt

from numba import njit

from .._dispatch import NPROP, make_state
from .._jit import JIT
from ..consts import KJ, R
from .gibbs import TAB, derivs, reduce

I, J, N, J0, N0 = TAB


@njit(**JIT)
def kernel(t, p):
    pi, tau = reduce(t, p)
    (g0, g0_tau, g0_tautau, gr,
     gr_pi, gr_pipi, gr_tau, gr_tautau, gr_pitau) = derivs(pi, tau, I, J, N, J0, N0)
    g_tau = g0_tau + gr_tau
    g_tautau = g0_tautau + gr_tautau
    pi_g_pi = 1.0 + pi * gr_pi          # π(γ⁰_π + γ^r_π), ибо γ⁰_π = 1/π
    num = (1.0 + pi * gr_pi - tau * pi * gr_pitau) ** 2
    den = 1.0 - pi * pi * gr_pipi
    rt = R * t
    v = rt * pi_g_pi / p / KJ
    return (v, 1.0 / v,
            rt * tau * g_tau,
            R * (tau * g_tau - (g0 + gr)),
            rt * (tau * g_tau - pi_g_pi),
            -R * tau * tau * g_tautau,
            R * (-tau * tau * g_tautau - num / den),
            sqrt(KJ * rt * (1.0 + 2.0 * pi * gr_pi
                            + pi * pi * gr_pi * gr_pi)
                 / (den + num / (tau * tau * g_tautau))))


@njit(**JIT)
def batch(t, p, out):
    for k in range(t.shape[0]):
        (out[0, k], out[1, k], out[2, k], out[3, k],
         out[4, k], out[5, k], out[6, k], out[7, k]) = kernel(t[k], p[k])


t_p = make_state(kernel, batch)
