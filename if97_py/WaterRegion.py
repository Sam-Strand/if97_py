from numba import float64, vectorize
from math import sqrt

I = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
     2, 3, 3, 3, 4, 4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32)
J = (-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -
     4, 0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41)
n = (0.14632971213167, -0.84548187169114, -0.37563603672040e1, 0.33855169168385e1, -0.95791963387872, 0.15772038513228, -0.16616417199501e-1, 0.81214629983568e-3, 0.28319080123804e-3, -0.60706301565874e-3, -0.18990068218419e-1, -0.32529748770505e-1, -0.21841717175414e-1, -0.52838357969930e-4, -0.47184321073267e-3, -0.30001780793026e-3, 0.47661393906987e-4, -
     0.44141845330846e-5, -0.72694996297594e-15, -0.31679644845054e-4, -0.28270797985312e-5, -0.85205128120103e-9, -0.22425281908000e-5, -0.65171222895601e-6, -0.14341729937924e-12, -0.40516996860117e-6, -0.12734301741641e-8, -0.17424871230634e-9, -0.68762131295531e-18, 0.14478307828521e-19, 0.26335781662795e-22, -0.11947622640071e-22, 0.18228094581404e-23, -0.93537087292458e-25)


R = 0.461526


'''
Безразмерная свободная энергия Гиббса γ и ее производные γ_π, γ_ππ, γ_τ, γ_ττ, γ_πτ
'''


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γ(π, τ):
    result = 0.0
    for i in range(34):
        result += n[i] * (7.1 - π) ** I[i] * (τ - 1.222) ** J[i]
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γ_π(π, τ):
    result = 0.0
    for i in range(34):
        result -= n[i] * I[i] * (7.1 - π) ** (I[i] - 1) * (τ - 1.222) ** J[i]
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γ_ππ(π, τ):
    result = 0.0
    for i in range(34):
        result += n[i] * I[i] * (I[i] - 1) * \
            (7.1 - π) ** (I[i] - 2) * (τ - 1.222) ** J[i]
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γ_τ(π, τ):
    result = 0.0
    for i in range(34):
        result += n[i] * (7.1 - π) ** I[i] * J[i] * (τ - 1.222) ** (J[i] - 1)
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γ_ττ(π, τ):
    result = 0.0
    for i in range(34):
        result += n[i] * (7.1 - π) ** I[i] * J[i] * \
            (J[i] - 1) * (τ - 1.222) ** (J[i] - 2)
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γ_πτ(π, τ):
    result = 0.0
    for i in range(34):
        result -= n[i] * I[i] * (7.1 - π) ** (I[i] - 1) * \
            J[i] * (τ - 1.222) ** (J[i] - 1)
    return result


@vectorize([float64(float64)], nopython=True, cache=True)
def get_π(p):
    return p / 16.53


@vectorize([float64(float64)], nopython=True, cache=True)
def get_τ(t):
    return 1386 / t


class WaterRegion:
    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True, cache=True)
    def enthalpy_t_p(t, p):
        '''
        Удельная энтальпия по температуре и давлению [кДж/кг]
        '''
        π = get_π(p)
        τ = get_τ(t)
        γ_τ = get_γ_τ(π, τ)
        return 639.675036 * γ_τ

    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True, cache=True)
    def entropy_t_p(t, p):
        '''
        Удельная энтропия по температуре и давлению [кДж/кг∙K]
        '''
        π = get_π(p)
        τ = get_τ(t)
        γ = get_γ(π, τ)
        γ_τ = get_γ_τ(π, τ)
        return R * (τ * γ_τ - γ)

    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True, cache=True)
    def volume_t_p(t, p):
        '''
        Удельный объем по температуре и давлению [м³/кг]
        '''
        π = get_π(p)
        τ = get_τ(t)
        γ_π = get_γ_π(π, τ)
        return π * γ_π * R * t / p / 1000

    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True, cache=True)
    def soundSpeed_t_p(t, p):
        '''
        Удельный объем по температуре и давлению [м³/кг]
        '''
        π = get_π(p)
        τ = get_τ(t)
        γ_π = get_γ_π(π, τ)
        γ_πτ = get_γ_πτ(π, τ)
        γ_ττ = get_γ_ττ(π, τ)
        γ_ππ = get_γ_ππ(π, τ)
        return sqrt(
            R * t * 1000 * γ_π ** 2
            / (
                (γ_π - τ * γ_πτ) ** 2
                / (τ ** 2 * γ_ττ)
                - γ_ππ
            )
        )
