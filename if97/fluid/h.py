from . import Helmholtz
from ..vec import vec, Number
from ..consts import R


@vec(2)
def t_ρ(t: Number, ρ: Number) -> Number:
    '''
    Вычисляет удельную энтальпию сверхкритической воды по температуре и плотности.
    
    Parameters
    ----------
    t : Number
        Температура [K].
    p : Number
        Давление [МПа].
        
    Returns
    -------
    Number
        Удельная энтальпия [кДж/кг].
    '''
    τ = Helmholtz.get_τ(t)
    δ = Helmholtz.get_δ(ρ)
    φ_τ = Helmholtz.get_φ_τ(τ, δ)
    φ_δ = Helmholtz.get_φ_δ(τ, δ)
    return R * t * (τ * φ_τ + δ * φ_δ)
