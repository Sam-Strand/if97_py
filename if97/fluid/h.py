from if97.vec import vec
from if97.types import ArrayLike, FloatArray
from if97.consts import R
from . import Helmholtz


@vec(2)
def t_ρ(t: ArrayLike, ρ: ArrayLike) -> FloatArray:
    '''
    Вычисляет удельную энтальпию сверхкритической воды по температуре и плотности.
    
    Parameters
    ----------
    t : ArrayLike
        Температура [K].
    p : ArrayLike
        Давление [МПа].
        
    Returns
    -------
    FloatArray
        Удельная энтальпия [кДж/кг].
    '''
    τ = Helmholtz.get_τ(t)
    δ = Helmholtz.get_δ(ρ)
    φ_τ = Helmholtz.get_φ_τ(τ, δ)
    φ_δ = Helmholtz.get_φ_δ(τ, δ)
    return R * t * (τ * φ_τ + δ * φ_δ)
