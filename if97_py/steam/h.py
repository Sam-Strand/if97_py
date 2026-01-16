from if97_py.vec import vec
from if97_py.types import ArrayLike, FloatArray
from . import Gibbs


@vec(2)
def t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Удельная энтальпия по температуре и давлению [Дж/кг]
    Notes
    -----
    τ = 540.0 / t
    R = 0.461526
    h = R * τ * t * (γ0_τ + γr_τ) -> h = 249.22404 * (γ0_τ + γr_τ)
    '''
    τ = Gibbs.get_τ(t)
    π = p
    γ0_τ = Gibbs.get_γ0_τ(τ)
    γr_τ = Gibbs.get_γr_τ(π, τ)
    return 249.22404 * (γ0_τ + γr_τ)
