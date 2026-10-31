'''Удельный объём (перегретый пар, область 2) по температуре и давлению, [м³/кг].

Соотношение стандарта: v·p/(RT) = π(γ⁰_π + γ^r_π)
'''

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
    pi_g_pi = 1.0 + pi * gr_pi          # π(γ⁰_π + γ^r_π), ибо γ⁰_π = 1/π
    return R * t * pi_g_pi / p / KJ


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
