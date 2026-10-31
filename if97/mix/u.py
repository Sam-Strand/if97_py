'''Удельная внутренняя энергия пароводяной смеси (область 4), [кДж/кг].

В двухфазной области линейно интерполируются только экстенсивные свойства:
y = y' + x(y" - y').
'''

from numba import njit

from .._dispatch import make2
from .._jit import JIT_NAN
from ..fluid import tables
from ..saturation import p_t as sat_p
from .phases import states

# Таблицы заморожены: распаковка пяти массивов стоит 427 нс на вызов,
# а точечному вызову её нечем амортизировать.
OFF, PAR, VN, LI, LJ = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)


@njit(**JIT_NAN)
def kernel_px(p, x):
    _, _, _, a1, _, _, _, a2 = states(p, OFF, PAR, VN, LI, LJ)
    return a1 + x * (a2 - a1)


@njit(**JIT_NAN)
def kernel_tx(t, x):
    return kernel_px(sat_p(t), x)


@njit(**JIT_NAN)
def batch_px(p, x, out):
    for k in range(p.shape[0]):
        out[k] = kernel_px(p[k], x[k])


@njit(**JIT_NAN)
def batch_tx(t, x, out):
    for k in range(t.shape[0]):
        out[k] = kernel_tx(t[k], x[k])


#: по давлению [МПа] и степени сухости
p_x = make2(kernel_px, batch_px)

#: по температуре [K] и степени сухости
t_x = make2(kernel_tx, batch_tx)
