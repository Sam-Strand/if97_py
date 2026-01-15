from if97_py.vec import vec
from if97_py.consts import R
from . import Gibbs


@vec(2)
def t_p(t, p):
    '''
    Удельная энтропия по температуре и давлению [кДж/кг∙K]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ = Gibbs.get_γ(π, τ)
    γ_τ = Gibbs.get_γ_τ(π, τ)
    return R * (τ * γ_τ - γ)
