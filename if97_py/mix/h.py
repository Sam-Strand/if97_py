from if97_py.vec import vec
from if97_py.bounds import saturationPressure_t, saturationTemp_p
import if97_py.steam as steam
import if97_py.water as water
import if97_py.fluid as fluid
from if97_py.types import ArrayLike, FloatArray
from if97_py.consts import t3Min, p3Min


#@vec(2)
def p_t_x(t: ArrayLike, p: ArrayLike, x: ArrayLike) -> FloatArray:
    if p > p3Min:
        regs = fluid.v.get_subregions_at_sat(p)
        h_low = fluid.h.t_ρ(t, 1 / fluid.v.t_p(t, p, regs[0]))
        h_high = fluid.h.t_ρ(t, 1 / fluid.v.t_p(t, p, regs[1]))
    else:
        h_low = water.h.t_p(t, p)
        h_high = steam.h.t_p(t, p)
    return (h_high - h_low) * x + h_low


#@vec(2)
def p_x(p: ArrayLike, x: ArrayLike) -> FloatArray:
    t = saturationTemp_p(p)
    return p_t_x(t, p, x)


#@vec(2)
def t_x(t: ArrayLike, x: ArrayLike) -> FloatArray:
    p = saturationPressure_t(t)
    return p_t_x(t, p, x)
