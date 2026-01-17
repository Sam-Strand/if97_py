from if97_py.vec import vec
from if97_py.bounds import saturationPressure_t, saturationTemp_p
import if97_py.steam as steam
import if97_py.water as water
import if97_py.fluid as fluid
from if97_py.types import ArrayLike, FloatArray
from if97_py.consts import t3Min, p3Min

import if97_py.fluid.bounds as fluid_bounds

saturation_t_643 = saturationPressure_t(643.15) # 21.04336731897525
saturation_t_623 = saturationPressure_t(623.15) # 16.52916425260448

@vec(3)
def p_t_x(t: ArrayLike, p: ArrayLike, x: ArrayLike) -> FloatArray:
    if p > p3Min:
        if p <= 19.00881189173929:
            if p > 16.52916425260448:
                reg_low = 2
                reg_high = 19
            else:
                reg_low = 2
                reg_high = 2
        else:
            if p <= 20.5:
                reg_low = 18
                reg_high = 19
            else:
                
                if p <= saturation_t_643:
                    reg_low = 18
                    reg_high = 17
                else:    
                    # Самая сложная зона p > saturation_t_643
                    if p <= 21.90096265:
                        reg_low = 20
                        reg_high = 23
                    else:
                        if p <= 21.93161551:
                            reg_low = 20
                            reg_high = 25
                        else:
                            reg_low = 20 if saturationTemp_p(p) <= fluid_bounds._t_uv(p) else 24
                            reg_high = 25
        h_low = fluid.h.t_ρ(t, 1 / fluid.v.t_p_reg(t, p, reg_low))
        h_high = fluid.h.t_ρ(t, 1 / fluid.v.t_p_reg(t, p, reg_high))
    else:
        h_low = water.h.t_p(t, p)
        h_high = steam.h.t_p(t, p)
    return (h_high - h_low) * x + h_low


@vec(2)
def p_x(p: ArrayLike, x: ArrayLike) -> FloatArray:
    t = saturationTemp_p(p)
    return p_t_x(t, p, x)


@vec(2)
def t_x(t: ArrayLike, x: ArrayLike) -> FloatArray:
    p = saturationPressure_t(t)
    return p_t_x(t, p, x)
