from numba import float64, vectorize
import math

J0 = (0, 1, -5, -4, -3, -2, -1, 2, 3)
n0 = (-0.96927686500217E+01, 0.10086655968018E+02, -0.56087911283020E-02, 0.71452738081455E-01,
      -0.40710498223928E+00, 0.14240819171444E+01, -0.43839511319450E+01, -0.28408632460772E+00, 0.21268463753307E-01)
I = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 7,
      7, 7, 8, 8, 9, 10, 10, 10, 16, 16, 18, 20, 20, 20, 21, 22, 23, 24, 24, 24)
J = (0, 1, 2, 3, 6, 1, 2, 4, 7, 36, 0, 1, 3, 6, 35, 1, 2, 3, 7, 3, 16, 35, 0,
      11, 25, 8, 36, 13, 4, 10, 14, 29, 50, 57, 20, 35, 48, 21, 53, 39, 26, 40, 58)
n = (
    -0.0017731742473212999, -0.017834862292357999, -0.045996013696365003, -0.057581259083432, -0.050325278727930002,
    -3.3032641670203e-05, -0.00018948987516315, -0.0039392777243355001, -0.043797295650572998, -2.6674547914087001e-05,
    2.0481737692308999e-08, 4.3870667284435001e-07, -3.2277677238570002e-05, -0.0015033924542148, -0.040668253562648998,
    -7.8847309559367001e-10, 1.2790717852285001e-08, 4.8225372718507002e-07, 2.2922076337661001e-06, -1.6714766451061001e-11,
    -0.0021171472321354998, -23.895741934103999, -5.9059564324270004e-18, -1.2621808899101e-06, -0.038946842435739003,
    1.1256211360459e-11, -8.2311340897998004, 1.9809712802088e-08, 1.0406965210174e-19, -1.0234747095929e-13, -1.0018179379511e-09,
    -8.0882908646984998e-11, 0.10693031879409, -0.33662250574170999, 8.9185845355420999e-25, 3.0629316876231997e-13,
    -4.2002467698208001e-06, -5.9056029685639003e-26, 3.7826947613457002e-06, -1.2768608934681e-15, 7.3087610595061e-29,
    5.5414715350778001e-17, -9.4369707241209998e-07
)

R = 0.461526

'''
Часть идеального газ γ^0 безразмерной свободной энергии Гиббса в идеальном газе и ее производные
'''


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γ0(π, τ):
    result = math.log(π)
    for i in range(9):
        result += n0[i] * τ ** J0[i]
    return result


@vectorize([float64(float64)], nopython=True, cache=True)
def get_γ0_π(π):
    return 1 / π


@vectorize([float64(float64)], nopython=True, cache=True)
def get_γ0_ππ(π):
    return -1 / π ** 2


@vectorize([float64(float64)], nopython=True, cache=True)
def get_γ0_τ(τ):
    result = 0.0
    for i in range(9):
        result += n0[i] * J0[i] * τ ** (J0[i] - 1)
    return result


@vectorize([float64(float64)], nopython=True, cache=True)
def get_γ0_ττ(τ):
    result = 0.0
    for i in range(9):
        result += n0[i] * J0[i] * (J0[i] - 1) * τ ** (J0[i] - 2)
    return result


'''
Остаточная часть γ^r безразмерной свободной энергии Гиббса в идеальном газе и ее производные
'''


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γr(π, τ):
    result = 0.0
    for i in range(43):
        result += n[i] * π ** I[i] * (τ - 0.5) ** J[i]
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γr_π(π, τ):
    result = 0.0
    for i in range(43):
        result += n[i] * I[i] * π ** (I[i] - 1) *(τ - 0.5) ** J[i]
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γr_ππ(π, τ):
    result = 0.0
    for i in range(43):
        result += n[i] * I[i] * (I[i] - 1) * π ** (I[i] - 2) * (τ - 0.5) ** J[i]
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γr_τ(π, τ):
    result = 0.0
    for i in range(43):
        result += n[i] * π ** I[i] * J[i] * (τ - 0.5) ** (J[i] - 1)
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γr_ττ(π, τ):
    result = 0.0
    for i in range(43):
        result += n[i] * π ** I[i] * J[i] * (J[i] - 1) * (τ - 0.5) ** (J[i] - 2)
    return result


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def get_γr_πτ(π, τ):
    result = 0.0
    for i in range(len(n)):
        result += n[i] * I[i] * π ** (I[i] - 1) * J[i] * (τ - 0.5) ** (J[i] - 1)
    return result


@vectorize([float64(float64)], nopython=True, cache=True)
def get_τ(t):
    return 540.0 / t


class SteamRegion:
    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True, cache=True)
    def enthalpy_t_p(t, p):
        τ = get_τ(t)
        π = p
        γ0_τ = get_γ0_τ(τ)
        γr_τ = get_γr_τ(π, τ)
        return 249.22404 * (γ0_τ + γr_τ)
    
    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True, cache=True)
    def entropy_t_p(t, p):
        '''
        Удельная энтропия по температуре и давлению [Дж/кг∙K]
        '''
        π = p
        τ = get_τ(t)
        γ0_τ = get_γ0_τ(τ)
        γr_τ = get_γr_τ(π, τ)
        γ0 = get_γ0(π, τ)
        γr = get_γr(π, τ)
        return R * (τ * (γ0_τ + γr_τ) - (γ0 + γr))
    
    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True, cache=True)
    def volume_t_p(t, p):
        '''
        Удельный объем по температуре и давлению [м³/кг]
        '''
        π = p
        τ = get_τ(t)
        γ0_π = get_γ0_π(π)
        γr_π = get_γr_π(π, τ)
        return 461526e-9 * (γ0_π + γr_π) * t
    
    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True, cache=True)
    def soundSpeed_t_p(t, p):
        '''
        Скорость звука [м/с]
        '''
        π = p
        τ = get_τ(t)
        
        γ0_ττ = get_γ0_ττ(τ)

        γr_π = get_γr_π(π, τ)
        γr_πτ = get_γr_πτ(π, τ)
        γr_ππ = get_γr_ππ(π, τ)
        γr_ττ = get_γr_ττ(π, τ)
        return math.sqrt(
            R * t * 1000 * (1 + 2 * π * γr_π + π ** 2 * γr_π ** 2)
            / (
                1 - π ** 2 * γr_ππ + (1 + π * γr_π - τ * π * γr_πτ) ** 2
                / τ ** 2
                / (γ0_ττ + γr_ττ)
            )
        )
