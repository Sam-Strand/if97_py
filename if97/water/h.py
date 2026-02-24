from if97.vec import vec
from if97.types import ArrayLike, FloatArray
from if97.consts import R
from . import Gibbs


@vec(2)
def t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Вычисляет удельную энтальпию перегретого пара по температуре и давлению.
    
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
        
    Notes
    -----
    τ = 1386 / t
    R = 0.461526
    h = R * t * τ * γ_τ = 639.675036 * γ_τ
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ_τ = Gibbs.get_γ_τ(π, τ)
    return 639.675036 * γ_τ
