'''
Автоматический выбор области по (T, p).

Отдельный модуль, потому что он единственный тянет все области разом:
обращение к if97.water.h компилирует только область 1, а обращение к
if97.state_t_p — всё. Поэтому main.py подключает его лениво.

Таблицы области 3 заморожены, а буферы под лестницы степеней в пакетном
цикле выделяются один раз на весь массив: иначе каждая точка платила бы и
распаковку пяти массивов, и четыре аллокации.
'''

from numba import njit
from numpy import nan

from ._dispatch import make_prop, make_state
from ._jit import JIT_NAN
from .bounds import region
from .fluid import tables as f_tables
from .fluid.helmholtz import TAB as FTAB, scratch as fluid_scratch
from .fluid.state import kernel_core as fluid_core
from .fluid.subregion import scratch as v_scratch, v_pt_core
from .gas.state import kernel as gas_state
from .steam.gibbs import TAB as STAB, scratch as steam_scratch
from .steam.state import kernel_core as steam_core
from .water.gibbs import TAB as WTAB, scratch as water_scratch
from .water.state import kernel_core as water_core

OFF, PAR, VN, LI, LJ = (f_tables.OFF, f_tables.PAR, f_tables.VN,
                        f_tables.LI, f_tables.LJ)
WI, WJ, WN = WTAB
SI, SJ, SN, SJ0, SN0 = STAB
FI, FJ, FN = FTAB


@njit(**JIT_NAN)
def core(t, p, w_i, w_j, s_i, s_j, f_i, f_j, vx, vy):
    '''
    (v, rho, h, s, u, cp, cv, w) с автоматическим выбором области.

    NaN вне диапазона IF97 и на самой линии насыщения: пара (p, T) там
    вырождена и однофазного состояния не задаёт.
    '''
    r = region(t, p)
    if r == 1.0:
        return water_core(t, p, w_i, w_j, WI, WJ, WN)
    if r == 2.0:
        return steam_core(t, p, s_i, s_j, SI, SJ, SN, SJ0, SN0)
    if r == 3.0:
        rho = 1.0 / v_pt_core(t, p, OFF, PAR, VN, LI, LJ, vx, vy)
        return fluid_core(t, rho, f_i, f_j, FI, FJ, FN)
    if r == 5.0:
        return gas_state(t, p)
    return nan, nan, nan, nan, nan, nan, nan, nan


@njit(**JIT_NAN)
def kernel(t, p):
    w_i, w_j = water_scratch()
    s_i, s_j = steam_scratch()
    f_i, f_j = fluid_scratch()
    vx, vy = v_scratch()
    return core(t, p, w_i, w_j, s_i, s_j, f_i, f_j, vx, vy)


@njit(**JIT_NAN)
def batch(t, p, out):
    w_i, w_j = water_scratch()      # один раз на весь массив, не на точку
    s_i, s_j = steam_scratch()
    f_i, f_j = fluid_scratch()
    vx, vy = v_scratch()
    for k in range(t.shape[0]):
        (out[0, k], out[1, k], out[2, k], out[3, k],
         out[4, k], out[5, k], out[6, k], out[7, k]) = core(
            t[k], p[k], w_i, w_j, s_i, s_j, f_i, f_j, vx, vy)


state_t_p = make_state(kernel, batch)
v_t_p = make_prop(kernel, batch, 'v')
rho_t_p = make_prop(kernel, batch, 'rho')
h_t_p = make_prop(kernel, batch, 'h')
s_t_p = make_prop(kernel, batch, 's')
u_t_p = make_prop(kernel, batch, 'u')
cp_t_p = make_prop(kernel, batch, 'cp')
cv_t_p = make_prop(kernel, batch, 'cv')
w_t_p = make_prop(kernel, batch, 'w')
