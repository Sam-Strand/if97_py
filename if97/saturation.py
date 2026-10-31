'''
Линия насыщения (область 4), уравнения (30) и (31),
коэффициенты — Таблица 34.

Лежит на верхнем уровне рядом с b23: это такая же линия на диаграмме.
Зависит только от констант, поэтому её могут импортировать и границы
областей, и выбор подобласти области 3, без циклов.
'''

from math import sqrt

from numba import njit
from numpy import nan

from ._jit import JIT_NAN
from .consts import PCRIT, TCRIT, minP, minT

N = (
    0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
    0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
    -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
    0.65017534844798e3,
)


@njit(**JIT_NAN)
def p_t(t):
    '''Давление насыщения по температуре, МПа. NaN вне 273.15..647.096 K.'''
    if t != t or t < minT or t > TCRIT:   # t != t ловит NaN
        return nan
    th = t + N[8] / (t - N[9])
    a = th * th + N[0] * th + N[1]
    b = N[2] * th * th + N[3] * th + N[4]
    c = N[5] * th * th + N[6] * th + N[7]
    x = 2.0 * c / (-b + sqrt(b * b - 4.0 * a * c))
    x *= x
    return x * x


@njit(**JIT_NAN)
def t_p(p):
    '''Температура насыщения по давлению, K. NaN вне 611.213 Па..22.064 МПа.'''
    if p != p or p < minP or p > PCRIT:   # p != p ловит NaN
        return nan
    be = sqrt(sqrt(p))
    e = be * be + N[2] * be + N[5]
    f = N[0] * be * be + N[3] * be + N[6]
    g = N[1] * be * be + N[4] * be + N[7]
    d = 2.0 * g / (-f - sqrt(f * f - 4.0 * e * g))
    q = N[9] + d
    return (q - sqrt(q * q - 4.0 * (N[8] + N[9] * d))) / 2.0


@njit(**JIT_NAN)
def p_t_batch(t, out):
    for k in range(t.shape[0]):
        out[k] = p_t(t[k])


@njit(**JIT_NAN)
def t_p_batch(p, out):
    for k in range(p.shape[0]):
        out[k] = t_p(p[k])
