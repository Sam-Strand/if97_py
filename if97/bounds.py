'''
Границы областей IAPWS-IF97 и определение области по (T, p).

Импортирует только пограничные уравнения — линию насыщения, B23 и B2bc,
все три лёгкие. Уравнения состояния областей отсюда не тянутся, так что
`import if97.bounds` не запускает компиляцию полиномов.
'''

from numba import njit
from numpy import nan

from . import b23, b2bc, saturation
from ._dispatch import make1, make2
from ._jit import JIT_NAN
from .consts import (PCRIT, TCRIT, maxP, maxT, minP, minT, p2bcMin, p3Min,
                     p5Max, t3Min, t5Max)


@njit(**JIT_NAN)
def border_p(t):
    '''Давление на левой границе области 2: насыщение ниже t3Min, B23 выше.'''
    if t <= t3Min:
        return saturation.p_t(t)
    return b23.p_t(t)


@njit(**JIT_NAN)
def border_t(p):
    '''Температура на левой границе области 2: насыщение ниже p3Min, B23 выше.'''
    if p < p3Min:
        return saturation.t_p(p)
    return b23.t_p(p)


@njit(**JIT_NAN)
def region(t, p):
    '''
    Код области: 1 вода, 2 пар, 3 флюид, 4 линия насыщения,
    5 высокотемпературный пар, NaN вне диапазона IF97.

    Область 4 возвращается только при точном совпадении p с p_s(T). Пара (p, T)
    на линии насыщения вырождена и состояния не задаёт: для двухфазных нужна
    пара со степенью сухости, энтальпией или энтропией.
    '''
    if p != p or t != t or p < minP or t < minT:   # != ловит NaN
        return nan
    if t > maxT:
        # область 5: своя температурная полоса и свой потолок 50 МПа
        if t > t5Max or p > p5Max:
            return nan
        return 5.0
    if p > maxP:
        return nan
    if t <= t3Min:
        ps = saturation.p_t(t)
        if p == ps:
            return 4.0
        return 1.0 if p > ps else 2.0
    if t > 863.15:
        return 2.0
    return 3.0 if p > b23.p_t(t) else 2.0


@njit(**JIT_NAN)
def border_p_batch(t, out):
    for k in range(t.shape[0]):
        out[k] = border_p(t[k])


@njit(**JIT_NAN)
def border_t_batch(p, out):
    for k in range(p.shape[0]):
        out[k] = border_t(p[k])


@njit(**JIT_NAN)
def region_batch(t, p, out):
    for k in range(t.shape[0]):
        out[k] = region(t[k], p[k])


#: Давление насыщения по температуре [МПа].
saturationPressure_t = make1(saturation.p_t, saturation.p_t_batch)

#: Температура насыщения по давлению [K].
saturationTemp_p = make1(saturation.t_p, saturation.t_p_batch)

#: Давление на левой границе области 2 [МПа].
borderPressure_t = make1(border_p, border_p_batch)

#: Температура на левой границе области 2 [K].
borderTemp_p = make1(border_t, border_t_batch)

#: Граница областей 2 и 3 (B23): давление по температуре [МПа].
b23Pressure_t = make1(b23.p_t, b23.p_t_batch)

#: Граница областей 2 и 3 (B23): температура по давлению [K].
b23Temp_p = make1(b23.t_p, b23.t_p_batch)

#: Граница подобластей 2b и 2c (B2bc): энтальпия по давлению [кДж/кг].
b2bcEnthalpy_p = make1(b2bc.h_p_kernel, b2bc.h_p_batch)

#: Граница подобластей 2b и 2c (B2bc): давление по энтальпии [МПа].
b2bcPressure_h = make1(b2bc.p_h_kernel, b2bc.p_h_batch)

#: Код области по температуре и давлению.
region_t_p = make2(region, region_batch)

__all__ = ['saturationPressure_t', 'saturationTemp_p', 'borderPressure_t',
           'borderTemp_p', 'b23Pressure_t', 'b23Temp_p',
           'b2bcEnthalpy_p', 'b2bcPressure_h', 'region_t_p',
           'minP', 'maxP', 'minT', 'maxT', 't5Max', 'p5Max',
           'p3Min', 't3Min', 'p2bcMin', 'PCRIT', 'TCRIT']
