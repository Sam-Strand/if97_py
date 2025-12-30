from if97_py.vec import vec
from math import sqrt

from .. import Gibbs
R = 0.461526

@vec(2)
def enthalpy_t_p(t, p):
    τ = Gibbs.get_τ(t)
    π = p
    γ0_τ = Gibbs.get_γ0_τ(τ)
    γr_τ = Gibbs.get_γr_τ(π, τ)
    return 249.22404 * (γ0_τ + γr_τ)


@vec(2)
def entropy_t_p(t, p):
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


@vec(2)
def volume_t_p(t, p):
    '''
    Удельный объем по температуре и давлению [м³/кг]
    '''
    π = p
    τ = Gibbs.get_τ(t)
    γ0_π = Gibbs.get_γ0_π(π)
    γr_π = Gibbs.get_γr_π(π, τ)
    return 461526e-9 * (γ0_π + γr_π) * t


@vec(2)
def soundSpeed_t_p(t, p):
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
        R * t * 1000 * (1 + 2 * π * γr_π + π ** 2 * γr_π ** 2)
        / (
            1 - π ** 2 * γr_ππ + (1 + π * γr_π - τ * π * γr_πτ) ** 2
            / τ ** 2
            / (γ0_ττ + γr_ττ)
        )
    )