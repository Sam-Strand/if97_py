from if97.vec import vec
from if97.consts import R
from if97.types import ArrayLike, FloatArray
from . import Gibbs


@vec(2)
def t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Удельная энтропия по температуре и давлению [Дж/кг∙K]
    '''
    π = p
    τ = Gibbs.get_τ(t)
    γ0_τ = Gibbs.get_γ0_τ(τ)
    γr_τ = Gibbs.get_γr_τ(π, τ)
    γ0 = Gibbs.get_γ0(π, τ)
    γr = Gibbs.get_γr(π, τ)
    return R * (τ * (γ0_τ + γr_τ) - (γ0 + γr))
