'''
Расчет региона воды (1) по температуре и давлению
'''

from numba import float64, vectorize
from math import sqrt

from . import Gibbs
R = 0.461526

@vectorize([float64(float64, float64)], nopython=True, cache=True)
def enthalpy_t_p(t, p):
    '''
    Удельная энтальпия по температуре и давлению [кДж/кг]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ_τ = Gibbs.get_γ_τ(π, τ)
    return 639.675036 * γ_τ


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def entropy_t_p(t, p):
    '''
    Удельная энтропия по температуре и давлению [кДж/кг∙K]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ = Gibbs.get_γ(π, τ)
    γ_τ = Gibbs.get_γ_τ(π, τ)
    return R * (τ * γ_τ - γ)


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def volume_t_p(t, p):
    '''
    Удельный объем по температуре и давлению [м³/кг]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ_π = Gibbs.get_γ_π(π, τ)
    return π * γ_π * R * t / p / 1000


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def soundSpeed_t_p(t, p):
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
        R * t * 1000 * γ_π ** 2
        / (
            (γ_π - τ * γ_πτ) ** 2
            / (τ ** 2 * γ_ττ)
            - γ_ππ
        )
    )

@vectorize([float64(float64, float64)], nopython=True, cache=True)
def soundSpeed_t_p(t, p):
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
        R * t * 1000 * γ_π ** 2
        / (
            (γ_π - τ * γ_πτ) ** 2
            / (τ ** 2 * γ_ττ)
            - γ_ππ
        )
    )
