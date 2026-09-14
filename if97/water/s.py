from . import Gibbs
from ..vec import vec, Number
from ..consts import R


@vec(2)
def t_p(t: Number, p: Number) -> Number:
    '''
    Удельная энтропия по температуре и давлению [кДж/кг⋅K]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ = Gibbs.get_γ(π, τ)
    γ_τ = Gibbs.get_γ_τ(π, τ)
    return R * (τ * γ_τ - γ)
