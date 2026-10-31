'''
Состояния кипящей жидкости и сухого насыщенного пара.

Ниже p3Min = 16.5292 МПа насыщенные фазы лежат в областях 1 и 2. Выше — обе
в области 3, и объём каждой фазы берётся из обратного уравнения v(p, T) с
ЯВНО заданной подобластью: на самой линии насыщения выбор подобласти по (p, T)
вырожден, поэтому пара подобластей назначается по давлению (SR5-05).

Это место было источником ошибки в версии 1.1.2: энтальпия смеси учитывала
область 3, а энтропия — нет, и выше 16.53 МПа расходилась до 1.8e-3.
'''

from numba import njit
from numpy import nan

from .._jit import JIT_NAN
from ..consts import p3Min
from ..fluid import tables as f_tables
from ..fluid.state import kernel_trho as fluid_state
from ..fluid.subregion import scratch as v_scratch, v_reg
from ..saturation import t_p as sat_t
from ..steam.state import kernel as steam_state
from ..water.state import kernel as water_state


SAT_623 = 16.52916425260448   # p_s(623.15 K)
SAT_643 = 21.04336731897525   # p_s(643.15 K)


@njit(**JIT_NAN)
def _t_uv(p):
    return (0.528199646263062e3 + 0.890579602135307e1 * p
            - 0.222814134903755 * p * p + 0.286791682263697e-2 * p * p * p)


@njit(**JIT_NAN)
def sub_pair(p):
    '''Пара подобластей области 3 для кипящей жидкости и сухого пара.'''
    if p <= SAT_623:
        return 2, 2
    if p <= 19.00881189173929:
        return 2, 19
    if p <= 20.5:
        return 18, 19
    if p <= SAT_643:
        return 18, 17
    if p <= 21.90096265:
        return 20, 23
    if p <= 21.93161551:
        return 20, 25
    return (20 if sat_t(p) <= _t_uv(p) else 24), 25


@njit(**JIT_NAN)
def states(p, off, par, vn, li, lj):
    '''(v', h', s', u', v", h", s", u") при данном давлении; NaN вне области 4.'''
    t = sat_t(p)
    if t != t:
        return nan, nan, nan, nan, nan, nan, nan, nan
    if p > p3Min:
        lo, hi = sub_pair(p)
        pw_x, pw_y = v_scratch()
        v1 = v_reg(t, p, lo, off, par, vn, li, lj, pw_x, pw_y)
        v2 = v_reg(t, p, hi, off, par, vn, li, lj, pw_x, pw_y)
        _, _, h1, s1, u1, _, _, _ = fluid_state(t, 1.0 / v1)
        _, _, h2, s2, u2, _, _, _ = fluid_state(t, 1.0 / v2)
    else:
        v1, _, h1, s1, u1, _, _, _ = water_state(t, p)
        v2, _, h2, s2, u2, _, _, _ = steam_state(t, p)
    return v1, h1, s1, u1, v2, h2, s2, u2