from if97_py.vec import vec
from if97_py.consts import R
from . import Gibbs


@vec(2)
def t_p(t, p):
    '''
    Удельный объем по температуре и давлению [м³/кг]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ_π = Gibbs.get_γ_π(π, τ)
    return π * γ_π * R * t / p / 1000
