'''Плотность (сверхкритический флюид, область 3), [кг/м³].'''

from numba import njit

from .._dispatch import make2
from .._jit import JIT
from . import tables
from .subregion import v_pt, v_pt_core, scratch as v_scratch

# таблицы для пакетного цикла (аргументами — так ядро кэшируется)
_T = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)
OFF, PAR, VN, LI, LJ = _T

# они же для скалярного входа: заморожены, чтобы не распаковывать
# пять массивов на каждый вызов
OFF, PAR, VN, LI, LJ = _T


@njit(**JIT)
def kernel_tp(t, p):
    pw_x, pw_y = v_scratch()
    return 1.0 / v_pt_core(t, p, OFF, PAR, VN, LI, LJ, pw_x, pw_y)


@njit(**JIT)
def batch_tp(t, p, out, off, par, vn, li, lj):
    pw_x, pw_y = v_scratch()        # один раз на весь массив, не на точку
    for k in range(t.shape[0]):
        out[k] = 1.0 / v_pt_core(t[k], p[k], off, par, vn, li, lj, pw_x, pw_y)


@njit(**JIT)
def kernel_trho(t, rho):
    return rho


@njit(**JIT)
def batch_trho(t, rho, out):
    for k in range(t.shape[0]):
        out[k] = rho[k]


t_p = make2(kernel_tp, batch_tp, _T)
t_rho = make2(kernel_trho, batch_trho)
t_ρ = t_rho
