'''Удельный объём (сверхкритический флюид, область 3), [м³/кг].

По (T, p) берётся напрямую из обратного уравнения SR5-05; по (T, ρ) тривиально.
'''

from numba import njit

from .._dispatch import make2
from .._jit import JIT
from . import tables
from .subregion import scratch as v_scratch, v_pt_batch, v_pt_core

_T = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)
OFF, PAR, VN, LI, LJ = _T


@njit(**JIT)
def kernel_trho(t, rho):
    return 1.0 / rho


@njit(**JIT)
def batch_trho(t, rho, out):
    for k in range(t.shape[0]):
        out[k] = 1.0 / rho[k]


@njit(**JIT)
def kernel_tp(t, p):
    pw_x, pw_y = v_scratch()
    return v_pt_core(t, p, OFF, PAR, VN, LI, LJ, pw_x, pw_y)


t_p = make2(kernel_tp, v_pt_batch, _T)
t_rho = make2(kernel_trho, batch_trho)
t_ρ = t_rho
