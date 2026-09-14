from . import Gibbs
from ..vec import vec, Number


@vec(2)
def t_p(t: Number, p: Number) -> Number:
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
