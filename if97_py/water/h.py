from if97_py.vec import vec

from . import Gibbs

@vec(2)
def t_p(t, p):
    '''
    Удельная энтальпия по температуре и давлению [кДж/кг]
    '''
    π = Gibbs.get_π(p)
    τ = Gibbs.get_τ(t)
    γ_τ = Gibbs.get_γ_τ(π, τ)
    return 639.675036 * γ_τ