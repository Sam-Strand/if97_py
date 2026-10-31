'''Изохорная теплоёмкость (высокотемпературный пар, область 5) по температуре и давлению, [кДж/(кг·K)].

Соотношение стандарта: cv/R = -τ² γ_ττ - (1 + π γ^r_π - τπ γ^r_πτ)²/(1 - π² γ^r_ππ)
'''

from numba import njit

from .._dispatch import make2
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
    num = (1.0 + pi * gr_pi - tau * pi * gr_pitau) ** 2
    den = 1.0 - pi * pi * gr_pipi
    return R * (-tau * tau * g_tautau - num / den)


@njit(**JIT)
def batch(t, p, out):
    for k in range(t.shape[0]):
        out[k] = kernel(t[k], p[k])


#: Скаляр или массив; обе ветви зовут одно и то же ядро.
t_p = make2(kernel, batch)
