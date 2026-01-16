from math import sqrt
from if97_py.types import ArrayLike, FloatArray
from if97_py.vec import vec
from . import Gibbs


@vec(2)
def t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Скорость звука [м/с]
    '''
    π = p
    τ = Gibbs.get_τ(t)
    
    γ0_ττ = Gibbs.get_γ0_ττ(τ)
    γr_π = Gibbs.get_γr_π(π, τ)
    γr_πτ = Gibbs.get_γr_πτ(π, τ)
    γr_ππ = Gibbs.get_γr_ππ(π, τ)
    γr_ττ = Gibbs.get_γr_ττ(π, τ)
    return sqrt(
        461.526 * t * (1 + 2 * π * γr_π + π ** 2 * γr_π ** 2)
        / (
            1 - π ** 2 * γr_ππ + (1 + π * γr_π - τ * π * γr_πτ) ** 2
            / τ ** 2
            / (γ0_ττ + γr_ττ)
        )
    )
