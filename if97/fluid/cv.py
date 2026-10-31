'''Изохорная теплоёмкость (сверхкритический флюид, область 3), [кДж/(кг·K)].

Соотношение стандарта (Таблица 31): cv/R = -τ² φ_ττ
'''

from numba import njit

from .._dispatch import make2
from .._jit import JIT
from ..consts import KJ, R
from . import tables
from .helmholtz import TAB, derivs, reduce, scratch
from .subregion import v_pt, v_pt_core, scratch as v_scratch

# таблицы для пакетного цикла (аргументами — так ядро кэшируется)
I, J, N = TAB

_T = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)

# они же для скалярного входа: заморожены, чтобы не распаковывать
# пять массивов на каждый вызов
OFF, PAR, VN, LI, LJ = _T


@njit(**JIT)
def kernel_core(t, rho, pw_i, pw_j, I, J, N):
    tau, delta = reduce(t, rho)
    f, f_d, f_dd, f_tau, f_tautau, f_dtau = derivs(tau, delta, pw_i, pw_j, I, J, N)
    return -R * tau * tau * f_tautau


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


@njit(**JIT)
def kernel_tp(t, p):
    pw_i, pw_j = scratch()
    pw_x, pw_y = v_scratch()
    rho = 1.0 / v_pt_core(t, p, OFF, PAR, VN, LI, LJ, pw_x, pw_y)
    return kernel_core(t, rho, pw_i, pw_j, I, J, N)


@njit(**JIT)
def batch_tp(t, p, out, off, par, vn, li, lj):
    pw_i, pw_j = scratch()
    pw_x, pw_y = v_scratch()
    for k in range(t.shape[0]):
        rho = 1.0 / v_pt_core(t[k], p[k], off, par, vn, li, lj, pw_x, pw_y)
        out[k] = kernel_core(t[k], rho, pw_i, pw_j, I, J, N)


#: по температуре [K] и давлению [МПа], через обратное уравнение v(p, T)
t_p = make2(kernel_tp, batch_tp, _T)
