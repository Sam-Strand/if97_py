from if97.vec import vec
from if97.bounds import saturationPressure_t, saturationTemp_p
import if97.steam as steam
import if97.water as water
from if97.types import ArrayLike, FloatArray


@vec(2)
def t_x(t: ArrayLike, x: ArrayLike) -> FloatArray:
    p = saturationPressure_t(t)
    s1 = water.s.t_p(t, p)
    s2 = steam.s.t_p(t, p)
    return (s2 - s1) * x + s1


@vec(2)
def p_x(p: ArrayLike, x: ArrayLike) -> FloatArray:
    t = saturationTemp_p(p)
    s1 = water.s.t_p(t, p)
    s2 = steam.s.t_p(t, p)
    return (s2 - s1) * x + s1
