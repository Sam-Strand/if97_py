from if97_py.vec import vec
from if97_py.bounds import saturationPressure_t, saturationTemp_p
import if97_py.steam as steam
import if97_py.water as water


@vec(2)
def t_x(t, x):
    p = saturationPressure_t(t)
    h1 = water.h.t_p(t, p)
    h2 = steam.h.t_p(t, p)
    return (h2 - h1) * x + h1


@vec(2)
def p_x(p, x):
    t = saturationTemp_p(p)
    h1 = water.h.t_p(t, p)
    h2 = steam.h.t_p(t, p)
    return (h2 - h1) * x + h1
