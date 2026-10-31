'''
Состояние пароводяной смеси (область 4).

Возвращает namedtuple (v, rho, h, s, u). Изобарная и изохорная теплоёмкости
и скорость звука для смеси не определены: в двухфазной области линейно
интерполируются только экстенсивные свойства.
'''

from numba import njit

from .._dispatch import MixState, make_state
from .._jit import JIT_NAN
from ..fluid import tables
from ..saturation import p_t as sat_p
from .phases import states

OFF, PAR, VN, LI, LJ = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)


@njit(**JIT_NAN)
def kernel_px(p, x):
    v1, h1, s1, u1, v2, h2, s2, u2 = states(p, OFF, PAR, VN, LI, LJ)
    v = v1 + x * (v2 - v1)
    return (v, 1.0 / v, h1 + x * (h2 - h1),
            s1 + x * (s2 - s1), u1 + x * (u2 - u1))


@njit(**JIT_NAN)
def kernel_tx(t, x):
    return kernel_px(sat_p(t), x)


@njit(**JIT_NAN)
def batch_px(p, x, out):
    for k in range(p.shape[0]):
        (out[0, k], out[1, k], out[2, k],
         out[3, k], out[4, k]) = kernel_px(p[k], x[k])


@njit(**JIT_NAN)
def batch_tx(t, x, out):
    for k in range(t.shape[0]):
        (out[0, k], out[1, k], out[2, k],
         out[3, k], out[4, k]) = kernel_tx(t[k], x[k])


p_x = make_state(kernel_px, batch_px, cls=MixState)
t_x = make_state(kernel_tx, batch_tx, cls=MixState)
