'''
Полное состояние сверхкритического флюида (область 3).

Возвращает namedtuple (v, rho, h, s, u, cp, cv, w) за один обход таблицы.
Давление отдельно: по (T, ρ) оно само является искомой величиной, см. fluid.p.
'''

from math import sqrt

from numba import njit

from .._dispatch import make_state
from .._jit import JIT
from ..consts import KJ, R
from . import tables
from .helmholtz import TAB, derivs, reduce, scratch
from .subregion import v_pt, v_pt_core, scratch as v_scratch

# таблицы для пакетного цикла (аргументами — так ядро кэшируется)
I, J, N = TAB

_T = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)
OFF, PAR, VN, LI, LJ = _T

# они же для скалярного входа: заморожены, чтобы не распаковывать
# пять массивов на каждый вызов
OFF, PAR, VN, LI, LJ = _T


@njit(**JIT)
def kernel_core(t, rho, pw_i, pw_j, I, J, N):
    tau, delta = reduce(t, rho)
    f, f_d, f_dd, f_tau, f_tautau, f_dtau = derivs(tau, delta, pw_i, pw_j, I, J, N)
    rt = R * t
    d_fd = delta * f_d
    num = (d_fd - delta * tau * f_dtau) ** 2
    den = 2.0 * d_fd + delta * delta * f_dd
    return (1.0 / rho, rho,
            rt * (tau * f_tau + d_fd),
            R * (tau * f_tau - f),
            rt * tau * f_tau,
            R * (-tau * tau * f_tautau + num / den),
            -R * tau * tau * f_tautau,
            sqrt(KJ * rt * (den - num / (tau * tau * f_tautau))))


@njit(**JIT)
def kernel_trho(t, rho):
    pw_i, pw_j = scratch()
    return kernel_core(t, rho, pw_i, pw_j, I, J, N)


@njit(**JIT)
def kernel_tp(t, p):
    pw_i, pw_j = scratch()
    pw_x, pw_y = v_scratch()
    rho = 1.0 / v_pt_core(t, p, OFF, PAR, VN, LI, LJ, pw_x, pw_y)
    return kernel_core(t, rho, pw_i, pw_j, I, J, N)


@njit(**JIT)
def batch_trho(t, rho, out):
    pw_i, pw_j = scratch()
    for k in range(t.shape[0]):
        (out[0, k], out[1, k], out[2, k], out[3, k],
         out[4, k], out[5, k], out[6, k], out[7, k]) = kernel_core(
            t[k], rho[k], pw_i, pw_j, I, J, N)


@njit(**JIT)
def batch_tp(t, p, out, off, par, vn, li, lj):
    pw_i, pw_j = scratch()
    pw_x, pw_y = v_scratch()
    for k in range(t.shape[0]):
        rho = 1.0 / v_pt_core(t[k], p[k], off, par, vn, li, lj, pw_x, pw_y)
        (out[0, k], out[1, k], out[2, k], out[3, k],
         out[4, k], out[5, k], out[6, k], out[7, k]) = kernel_core(
            t[k], rho, pw_i, pw_j, I, J, N)


t_p = make_state(kernel_tp, batch_tp, _T)
t_rho = make_state(kernel_trho, batch_trho)
t_ρ = t_rho
