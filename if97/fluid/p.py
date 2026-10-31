'''Давление (сверхкритический флюид, область 3), [МПа].

Соотношение стандарта (Таблица 31): p/(ρRT) = δ φ_δ
'''

from numba import njit

from .._dispatch import make2
from .._jit import JIT
from ..consts import KJ, R
from . import tables
from .helmholtz import TAB, derivs, reduce, scratch


I, J, N = TAB

_T = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)


@njit(**JIT)
def kernel_core(t, rho, pw_i, pw_j, I, J, N):
    tau, delta = reduce(t, rho)
    f, f_d, f_dd, f_tau, f_tautau, f_dtau = derivs(tau, delta, pw_i, pw_j, I, J, N)
    d_fd = delta * f_d
    return rho * R * t * d_fd / KJ


@njit(**JIT)
def kernel_trho(t, rho):
    pw_i, pw_j = scratch()
    return kernel_core(t, rho, pw_i, pw_j, I, J, N)


@njit(**JIT)
def batch_trho(t, rho, out):
    pw_i, pw_j = scratch()          # один раз на весь массив, не на точку
    for k in range(t.shape[0]):
        out[k] = kernel_core(t[k], rho[k], pw_i, pw_j, I, J, N)


#: по температуре [K] и плотности [кг/м³]
t_rho = make2(kernel_trho, batch_trho)
t_ρ = t_rho
