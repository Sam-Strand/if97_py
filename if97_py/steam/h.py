from if97_py.vec import vec

from . import Gibbs

@vec(2)
def t_p(t, p):
    '''
    Удельная энтальпия по температуре и давлению [Дж/кг]
    '''
    τ = Gibbs.get_τ(t)
    π = p
    γ0_τ = Gibbs.get_γ0_τ(τ)
    γr_τ = Gibbs.get_γr_τ(π, τ)
    return 249.22404 * (γ0_τ + γr_τ)
