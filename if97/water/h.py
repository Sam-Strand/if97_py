from . import Gibbs
from ..vec import vec, Number


@vec(2)
def t_p(t: Number, p: Number) -> Number:
    '''
    Вычисляет удельную энтальпию перегретого пара по температуре и давлению.
    
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
