from if97.vec import vec
from if97.consts import R
from if97.types import ArrayLike, FloatArray
from . import Gibbs


@vec(2)
def t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Удельная энтропия по температуре и давлению [кДж/кг∙K]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ = Gibbs.get_γ(π, τ)
    γ_τ = Gibbs.get_γ_τ(π, τ)
    return R * (τ * γ_τ - γ)
