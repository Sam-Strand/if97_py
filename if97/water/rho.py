'''Плотность (вода, область 1) по температуре и давлению, [кг/м³].

Соотношение стандарта: ρ = 1/v
'''

from numba import njit

from .._dispatch import make2
from .._jit import JIT
from ..consts import KJ, R
from .gibbs import TAB, derivs, reduce, scratch

I, J, N = TAB


@njit(**JIT)
def kernel_core(t, p, pw_i, pw_j, I, J, N):
    pi, tau = reduce(t, p)
    g, g_pi, g_pipi, g_tau, g_tautau, g_pitau = derivs(pi, tau, pw_i, pw_j, I, J, N)
    return p * KJ / (R * t * pi * g_pi)


@njit(**JIT)
def kernel(t, p):
    pw_i, pw_j = scratch()
    return kernel_core(t, p, pw_i, pw_j, I, J, N)


@njit(**JIT)
def batch(t, p, out):
    pw_i, pw_j = scratch()          # один раз на весь массив, не на точку
    for k in range(t.shape[0]):
        out[k] = kernel_core(t[k], p[k], pw_i, pw_j, I, J, N)


#: Скаляр или массив; обе ветви зовут одно и то же ядро.
t_p = make2(kernel, batch)
