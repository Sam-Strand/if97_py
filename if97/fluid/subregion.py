'''
Обратное уравнение v(p, T) области 3 по IAPWS SR5-05.

Область 3 разбита на 26 подобластей 3a..3z; `of_tp` выбирает подобласть по
давлению и температуре, `v_reg` считает объём в заданной подобласти, `v_pt`
соединяет их. Границы подобластей — вспомогательные полиномы T(p) ниже.

Таблицы передаются аргументами, а не читаются как глобальные массивы: numba
отключает cache=True для функций, использующих большие глобальные массивы,
а это самое дорогое для компиляции место в пакете.
'''

from math import exp, log

from numba import njit

from .._dispatch import make2
from .._jit import JIT, JIT_NAN
from ..consts import PCRIT, RHOCRIT, TCRIT

PWLEN = 49   # 36 - (-12) + 1: полный диапазон показателей SR5-05
from .. import saturation
from . import tables

import numpy as np

@njit(**JIT_NAN)
def _t_cd(p):
    return (0.585276966696349e3 + 0.278233532206915e1 * p
            - 0.127283549295878e-1 * p * p + 0.159090746562729e-3 * p * p * p)


@njit(**JIT_NAN)
def _t_gh(p):
    return (-0.249284240900418e5 + 0.428143584791546e4 * p
            - 0.269029173140130e3 * p ** 2 + 0.751608051114157e1 * p ** 3
            - 0.787105249910383e-1 * p ** 4)


@njit(**JIT_NAN)
def _t_ij(p):
    return (0.584814781649163e3 - 0.616179320924617 * p
            + 0.260763050899562 * p ** 2 - 0.587071076864459e-2 * p ** 3
            + 0.515308185433082e-4 * p ** 4)


@njit(**JIT_NAN)
def _t_jk(p):
    return (0.617229772068439e3 - 0.770600270141675e1 * p
            + 0.697072596851896 * p ** 2 - 0.157391839848015e-1 * p ** 3
            + 0.137897492684194e-3 * p ** 4)


@njit(**JIT_NAN)
def _t_mn(p):
    return (0.535339483742384e3 + 0.761978122720128e1 * p
            - 0.158365725441648 * p * p + 0.192871054508108e-2 * p * p * p)


@njit(**JIT_NAN)
def _t_qu(p):
    return (0.565603648239126e3 + 0.529062258221222e1 * p
            - 0.102020639611016 * p * p + 0.122240301070145e-2 * p * p * p)


@njit(**JIT_NAN)
def _t_rx(p):
    return (0.584561202520006e3 - 0.102961025163669e1 * p
            + 0.243293362700452 * p * p - 0.294905044740799e-2 * p * p * p)


@njit(**JIT_NAN)
def _t_uv(p):
    return (0.528199646263062e3 + 0.890579602135307e1 * p
            - 0.222814134903755 * p * p + 0.286791682263697e-2 * p * p * p)


@njit(**JIT_NAN)
def _t_ab(p):
    lp = log(p)
    return (0.154793642129415e4 - 0.187661219490113e3 * lp
            + 0.213144632222113e2 * lp * lp - 0.191887498864292e4 / lp
            + 0.918419702359447e3 / (lp * lp))


@njit(**JIT_NAN)
def _t_wx(p):
    lp = log(p)
    return (0.728052609145380e1 + 0.973505869861952e2 * lp
            + 0.147370491183191e2 * lp * lp + 0.329196213998375e3 / lp
            + 0.873371668682417e3 / (lp * lp))


@njit(**JIT_NAN)
def _t_ef(p):
    return 3.727888004 * (p - PCRIT) + TCRIT


@njit(**JIT_NAN)
def _t_op(p):
    lp = log(p)
    return (0.969461372400213e3 - 0.332500170441278e3 * lp
            + 0.642859598466067e2 * lp * lp + 0.773845935768222e3 / lp
            - 0.152313732937084e4 / (lp * lp))


SAT_643 = 21.04336731897525   # sat_p(643.15)
SAT_623 = 16.52916425260448   # sat_p(623.15)


@njit(**JIT_NAN)
def of_tp(t, p):
    '''Номер подобласти 3a..3z (0..25).'''
    if p > 40:
        return 0 if t <= _t_ab(p) else 1
    if p > 25:
        if t <= _t_cd(p):
            return 2
        if t <= _t_ab(p):
            return 3
        if t <= _t_ef(p):
            return 4
        return 5
    if p > 23.5:
        if t <= _t_cd(p):
            return 2
        if t <= _t_gh(p):
            return 6
        if t <= _t_ef(p):
            return 7
        if t <= _t_ij(p):
            return 8
        if t <= _t_jk(p):
            return 9
        return 10
    if p > 23:
        if t <= _t_cd(p):
            return 2
        if t <= _t_gh(p):
            return 11
        if t <= _t_ef(p):
            return 7
        if t <= _t_ij(p):
            return 8
        if t <= _t_jk(p):
            return 9
        return 10
    if p > 22.5:
        if t <= _t_cd(p):
            return 2
        if t <= _t_gh(p):
            return 11
        if t <= _t_mn(p):
            return 12
        if t <= _t_ef(p):
            return 13
        if t <= _t_op(p):
            return 14
        if t <= _t_ij(p):
            return 15
        if t <= _t_jk(p):
            return 9
        return 10
    if p > 20.5:
        if p > SAT_643:
            if t <= _t_cd(p):
                return 2
            if t <= _t_qu(p):
                return 16
            if t <= _t_rx(p):
                if p > 22.11:
                    if t <= _t_uv(p):
                        return 20
                    if t <= _t_ef(p):
                        return 21
                    if t <= _t_wx(p):
                        return 22
                    return 23
                if p > PCRIT:
                    if t <= _t_uv(p):
                        return 20
                    if t <= _t_ef(p):
                        return 24
                    if t <= _t_wx(p):
                        return 25
                    return 23
                if t > saturation.t_p(p):
                    if p > 21.90096265:
                        return 25 if t <= _t_wx(p) else 23
                    return 23
                if p <= 21.93161551:
                    return 20
                return 20 if t <= _t_uv(p) else 24
            if t <= _t_jk(p):
                return 17
            return 10
        if t <= _t_cd(p):
            return 2
        if t <= saturation.t_p(p):
            return 18
        if t <= _t_jk(p):
            return 17
        return 10
    if p > 19.00881189173929:
        if t <= _t_cd(p):
            return 2
        if t <= saturation.t_p(p):
            return 18
        return 19
    if p > SAT_623:
        return 2 if t <= saturation.t_p(p) else 19
    return 2


@njit(**JIT)
def scratch():
    '''Буферы под лестницы степеней; в пакетном режиме выделяются один раз.'''
    return np.empty(PWLEN), np.empty(PWLEN)


@njit(**JIT)
def v_reg(t, p, reg, off, par, vn, li, lj, pw_x, pw_y):
    '''Удельный объём в заданной подобласти области 3, м³/кг.'''
    if t == TCRIT and p == PCRIT:
        return 1.0 / RHOCRIT
    a0 = off[reg]
    a1 = off[reg + 1]
    vs = par[reg, 0]
    x = p / par[reg, 1] - par[reg, 3]
    y = t / par[reg, 2] - par[reg, 4]
    c = par[reg, 5]
    d = par[reg, 6]
    e = par[reg, 7]
    s = 0.0

    if c != 1.0 or d != 1.0:
        # 3n (экспоненциальная форма) и подобласти с дробными c или d:
        # лестницу на дробных показателях не построить
        if reg == 13:
            for k in range(a0, a1):
                s += vn[k] * x ** li[k] * y ** lj[k]
            return vs * exp(s)
        for k in range(a0, a1):
            s += vn[k] * x ** (c * li[k]) * y ** (d * lj[k])
        if e == 1.0:
            return vs * s
        s = s * s
        return vs * s * s

    # 19 подобластей из 26: целые показатели, идёт лестница
    ilo = int(par[reg, 8])
    ihi = int(par[reg, 9])
    jlo = int(par[reg, 10])
    jhi = int(par[reg, 11])
    pw_x[0] = x ** ilo          # один pow на ось, дальше умножения
    for m in range(1, ihi - ilo + 1):
        pw_x[m] = pw_x[m - 1] * x
    pw_y[0] = y ** jlo
    for m in range(1, jhi - jlo + 1):
        pw_y[m] = pw_y[m - 1] * y
    for k in range(a0, a1):
        s += vn[k] * pw_x[li[k] - ilo] * pw_y[lj[k] - jlo]
    if e == 1.0:
        return vs * s
    if e == 4.0:
        s = s * s
        return vs * s * s
    return vs * s ** e


@njit(**JIT)
def v_pt_core(t, p, off, par, vn, li, lj, pw_x, pw_y):
    '''v(p, T) при уже выделенных буферах — вход для пакетных циклов.'''
    return v_reg(t, p, of_tp(t, p), off, par, vn, li, lj, pw_x, pw_y)


@njit(**JIT)
def v_pt(t, p, off, par, vn, li, lj):
    '''Удельный объём по температуре и давлению, м³/кг.'''
    pw_x, pw_y = scratch()
    return v_pt_core(t, p, off, par, vn, li, lj, pw_x, pw_y)


_T = (tables.OFF, tables.PAR, tables.VN, tables.LI, tables.LJ)


@njit(**JIT_NAN)
def of_tp_batch(t, p, out):
    for k in range(t.shape[0]):
        out[k] = of_tp(t[k], p[k])


@njit(**JIT)
def v_pt_batch(t, p, out, off, par, vn, li, lj):
    pw_x, pw_y = scratch()
    for k in range(t.shape[0]):
        out[k] = v_pt_core(t[k], p[k], off, par, vn, li, lj, pw_x, pw_y)


#: Номер подобласти 0..25 (3a..3z).
get_sub_region = make2(of_tp, of_tp_batch, dtype=np.int64)
