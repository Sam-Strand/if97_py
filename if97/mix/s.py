from ..vec import vec, Number
from ..bounds import saturationPressure_t, saturationTemp_p
from .. import steam
from .. import water


@vec(2)
def t_x(t: Number, x: Number) -> Number:
    p = saturationPressure_t(t)
    s1 = water.s.t_p(t, p)
    s2 = steam.s.t_p(t, p)
    return (s2 - s1) * x + s1


@vec(2)
def p_x(p: Number, x: Number) -> Number:
    t = saturationTemp_p(p)
    s1 = water.s.t_p(t, p)
    s2 = steam.s.t_p(t, p)
    return (s2 - s1) * x + s1
