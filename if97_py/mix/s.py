from if97_py.vec import vec
from if97_py.bounds import saturationPressure_t, saturationTemp_p
import if97_py.steam as steam
import if97_py.water as water


@vec(2)
def t_x(t, x):
    p = saturationPressure_t(t)
    s1 = water.s.t_p(t, p)
    s2 = steam.s.t_p(t, p)
    return (s2 - s1) * x + s1


@vec(2)
def p_x(p, x):
    t = saturationTemp_p(p)
    s1 = water.s.t_p(t, p)
    s2 = steam.s.t_p(t, p)
    return (s2 - s1) * x + s1
