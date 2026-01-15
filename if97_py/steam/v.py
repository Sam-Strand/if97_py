from if97_py.vec import vec
from . import Gibbs


@vec(2)
def t_p(t, p):
    '''
    Удельный объем по температуре и давлению [м³/кг]
    '''
    π = p
    τ = Gibbs.get_τ(t)
    γ0_π = Gibbs.get_γ0_π(π)
    γr_π = Gibbs.get_γr_π(π, τ)
    return 461526e-9 * (γ0_π + γr_π) * t
