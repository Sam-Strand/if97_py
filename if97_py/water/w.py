from math import sqrt
from if97_py.types import ArrayLike, FloatArray
from if97_py.vec import vec
from if97_py.consts import R
from . import Gibbs


@vec(2)
def t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Удельный объем по температуре и давлению [м³/кг]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ_π = Gibbs.get_γ_π(π, τ)
    γ_πτ = Gibbs.get_γ_πτ(π, τ)
    γ_ττ = Gibbs.get_γ_ττ(π, τ)
    γ_ππ = Gibbs.get_γ_ππ(π, τ)
    return sqrt(
        461.526 * t * γ_π ** 2
        / (
            (γ_π - τ * γ_πτ) ** 2
            / (τ ** 2 * γ_ττ)
            - γ_ππ
        )
    )
