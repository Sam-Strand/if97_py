from if97_py.vec import vec
from if97_py.types import ArrayLike, FloatArray
from . import Gibbs


@vec(2)
def t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Удельный объем по температуре и давлению [м³/кг]
    Notes
    -----
    π = p / 16.53
    R = 0.461526
    v = π * R * t * γ_π / p / 1000 = 2.7920508166969e-05 * t * γ_π
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ_π = Gibbs.get_γ_π(π, τ)
    return 2.7920508166969e-05 * t * γ_π
