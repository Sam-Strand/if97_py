'''
Степень сухости пароводяной смеси.

Возвращает 0 при h <= h' и 1 при h >= h" (то же по энтропии), NaN если линия
насыщения при данном аргументе не определена — выше критической температуры
или критического давления двухфазного состояния не существует.
'''

from numba import njit
from numpy import nan

from .._dispatch import make2
from .._jit import JIT_NAN
from ..fluid import tables
from ..saturation import p_t as sat_p
from .phases import states

# таблицы заморожены: пять массивов в сигнатуре стоят 427 нс на вызов
OFF, PAR, VN, LI, LJ = (tables.OFF, tables.PAR, tables.VN,
                        tables.LI, tables.LJ)


@njit(**JIT_NAN)
def kernel_ph(p, h):
    _, h1, _, _, _, h2, _, _ = states(p, OFF, PAR, VN, LI, LJ)
    if h1 != h1:
        return nan
    if h <= h1:
        return 0.0
    if h >= h2:
        return 1.0
    return (h - h1) / (h2 - h1)


@njit(**JIT_NAN)
def kernel_ps(p, s):
    _, _, s1, _, _, _, s2, _ = states(p, OFF, PAR, VN, LI, LJ)
    if s1 != s1:
        return nan
    if s <= s1:
        return 0.0
    if s >= s2:
        return 1.0
    return (s - s1) / (s2 - s1)


@njit(**JIT_NAN)
def kernel_th(t, h):
    return kernel_ph(sat_p(t), h)


@njit(**JIT_NAN)
def kernel_ts(t, s):
    return kernel_ps(sat_p(t), s)


def _loop(kernel):
    @njit(**JIT_NAN)
    def batch(a, b, out):
        for k in range(a.shape[0]):
            out[k] = kernel(a[k], b[k])
    return batch


batch_ph = _loop(kernel_ph)
batch_ps = _loop(kernel_ps)
batch_th = _loop(kernel_th)
batch_ts = _loop(kernel_ts)

p_h = make2(kernel_ph, batch_ph)
p_s = make2(kernel_ps, batch_ps)
t_h = make2(kernel_th, batch_th)
t_s = make2(kernel_ts, batch_ts)
