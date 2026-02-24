from if97.vec import vec
from if97.types import ArrayLike, FloatArray
from math import log


n_1 = 1.0658070028513

I = (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 9, 9, 10, 10, 11)
J = (0, 1, 2, 7, 10, 12, 23, 2, 6, 15, 17, 0, 2, 6, 7, 22, 26, 0, 2, 4, 16, 26, 0, 2, 4, 26, 1, 3, 26, 0, 2, 26, 2, 26, 2, 26, 0, 1, 26)
n = (-0.15732845290239e2, 0.20944396974307e2, -0.76867707878716e1, 0.26185947787954e1, -0.28080781148620e1, 0.12053369696517e1, -0.84566812812502e-2, -0.12654315477714e1, -0.11524407806681e1, 0.88521043984318, -0.64207765181607, 0.38493460186671, -0.85214708824206, 0.48972281541877e1, -0.30502617256965e1, 0.39420536879154e-1, 0.12558408424308, -0.27999329698710, 0.13899799569460e1, -0.20189915023570e1, -0.82147637173963e-2, -0.47596035734923, 0.43984074473500e-1, -0.44476435428739, 0.90572070719733, .70522450087967, .10770512626332, -0.32913623258954, -0.50871062041158, -0.22175400873096e-1, 0.94260751665092e-1, 0.16436278447961, -0.13503372241348e-1, -0.14834345352472e-1, 0.57922953628084e-3, 0.32308904703711e-2, 0.80964802996215e-4, -0.16557679795037e-3, -0.44923899061815e-4)


@vec(2)
def get_φ(τ: ArrayLike, δ: ArrayLike) -> FloatArray:
    '''
    Безразмерная свободная энергия Гельмгольца φ(τ, δ) для области 3.
    
    Parameters
    ----------
    τ : ArrayLike
        Обратная приведенная температура τ = T*/t.
    δ : ArrayLike  
        Приведенная плотность δ = ρ/ρ*.
        
    Returns
    -------
    FloatArray
        φ(τ, δ).
    '''
    result = n_1 * log(δ)
    for i in range(39):
        result += n[i] * δ ** I[i] * τ ** J[i]
    return result


@vec(2)
def get_φ_δ(τ: ArrayLike, δ: ArrayLike) -> FloatArray:
    '''
    Первая производная свободной энергии Гельмгольца по δ: φ_δ(τ, δ).
    
    Parameters
    ----------
    τ : ArrayLike
        Обратная приведенная температура τ = T*/t.
    δ : ArrayLike  
        Приведенная плотность δ = ρ/ρ*.
        
    Returns
    -------
    FloatArray
        ∂φ/∂δ.
    '''
    result = n_1 / δ
    for i in range(39):
        result += n[i] * I[i] * δ ** (I[i] - 1) * τ ** J[i]
    return result


@vec(2)
def get_φ_δδ(τ: ArrayLike, δ: ArrayLike) -> FloatArray:
    '''
    Вторая производная свободной энергии Гельмгольца по δ: φ_δδ(τ, δ).
    
    Parameters
    ----------
    τ : ArrayLike
        Обратная приведенная температура τ = T*/t.
    δ : ArrayLike  
        Приведенная плотность δ = ρ/ρ*.
        
    Returns
    -------
    FloatArray
        ∂²φ/∂δ².
    '''
    result = -n_1 / δ ** 2
    for i in range(39):
        result += n[i] * I[i] * (I[i] - 1) * δ ** (I[i] - 2) * τ ** J[i]
    return result


@vec(2)
def get_φ_τ(τ: ArrayLike, δ: ArrayLike) -> FloatArray:
    '''
    Первая производная свободной энергии Гельмгольца по τ: φ_τ(τ, δ).
    
    Parameters
    ----------
    τ : ArrayLike
        Обратная приведенная температура τ = T*/t.
    δ : ArrayLike  
        Приведенная плотность δ = ρ/ρ*.
        
    Returns
    -------
    FloatArray
        ∂φ/∂τ.
    '''
    result = 0.0
    for i in range(39):
        result += n[i] * δ ** I[i] * J[i] * τ ** (J[i] - 1)
    return result


@vec(2)
def get_φ_ττ(τ: ArrayLike, δ: ArrayLike) -> FloatArray:
    '''
    Вторая производная свободной энергии Гельмгольца по τ: φ_ττ(τ, δ).
    
    Parameters
    ----------
    τ : ArrayLike
        Обратная приведенная температура τ = T*/t.
    δ : ArrayLike  
        Приведенная плотность δ = ρ/ρ*.
        
    Returns
    -------
    FloatArray
        ∂²φ/∂τ².
    '''
    result = 0.0
    for i in range(39):
        result += n[i] * δ ** I[i] * J[i] * (J[i] - 1) * τ ** (J[i] - 2)
    return result


@vec(2)
def get_φ_δτ(τ: ArrayLike, δ: ArrayLike) -> FloatArray:
    '''
    Смешанная производная свободной энергии Гельмгольца: φ_δτ(τ, δ).
    
    Parameters
    ----------
    τ : ArrayLike
        Обратная приведенная температура τ = T*/t.
    δ : ArrayLike  
        Приведенная плотность δ = ρ/ρ*.
    
    Returns
    -------
    FloatArray
        ∂²φ/∂δ∂τ.
    '''
    result = 0.0
    for i in range(39):
        result += n[i] * I[i] * δ ** (I[i] - 1) * J[i] * τ ** (J[i] - 1)
    return result


@vec(1)
def get_τ(t: ArrayLike) -> FloatArray:
    '''
    Вычисляет τ = T*/t.
    
    Parameters
    ----------
    t : ArrayLike
        Температура [K].
        
    Returns
    -------
    FloatArray
        τ.
    '''
    return 647.096 / t


@vec(1)
def get_δ(ρ: ArrayLike) -> FloatArray:
    '''
    Вычисляет δ = ρ/ρ*.
    
    Parameters
    ----------
    ρ : ArrayLike
        Плотность [кг/м³].
        
    Returns
    -------
    FloatArray
        δ.
    '''
    return ρ / 322.0
