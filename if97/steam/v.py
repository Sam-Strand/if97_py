from if97.vec import vec
from if97.types import ArrayLike, FloatArray
from . import Gibbs


@vec(2)
def t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Удельный объем по температуре и давлению [м³/кг]
    Notes
    -----
    π = p / 1
    R = 0.461526
    h = R * π * t *(γ0_π + γr_π) / 1000 / p -> h = 0.000461526 * t * (γ0_π + γr_π)
    '''
    π = p
    τ = Gibbs.get_τ(t)
    γ0_π = Gibbs.get_γ0_π(π)
    γr_π = Gibbs.get_γr_π(π, τ)
    return 0.000461526 * t * (γ0_π + γr_π)
