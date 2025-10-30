import numpy as np
from numba import float64, vectorize

from .SteamRegion import SteamRegion 
from .WaterRegion import WaterRegion


@vectorize([float64(float64)], nopython=True, cache=True)
def _saturationPressureT_calc(t):
    K1 = t - 0.23855557567849 / (t - 650.17534844798)
    K2 = -17.073846940092 * K1**2 + 12020.82470247 * K1 - 3232555.0322333
    K3 = 14.91510861353 * K1**2 - 4823.2657361591 * K1 + 405113.40542057
    
    discriminant = K2**2 - 4 * (K1**2 + 1167.0521452767 * K1 - 724213.16703206) * K3
    denominator = -K2 + np.sqrt(discriminant)
    
    return (2 * K3 / denominator)**4


@vectorize([float64(float64)], nopython=True, cache=True)
def _saturationTempP_calc(p):
    k1 = 650.17534844798
    p_quarter = p**0.25
    
    K1 = 1167.0521452767 * p_quarter**2 + 12020.82470247 * p_quarter - 4823.2657361591
    K2 = -724213.16703206 * p_quarter**2 - 3232555.0322333 * p_quarter + 405113.40542057
    discriminant = K1**2 - 4 * (p_quarter**2 - 17.073846940092 * p_quarter + 14.91510861353) * K2
    K3 = 2 * K2 / (-K1 - np.sqrt(discriminant))
    
    temp_k = (k1 + K3 - np.sqrt((k1 + K3)**2 - 4 * (k1 * K3 - 0.23855557567849))) / 2
    return temp_k


@vectorize([float64(float64)], nopython=True, cache=True)
def _borderPressureT_calc(t):
    if t <= 623.15:
        return _saturationPressureT_calc(t)
    else:
        return (348.05185628969 - 1.1671859879975 * t + 1.0192970039326e-3 * t**2)


@vectorize([float64(float64)], nopython=True, cache=True)
def _borderTempP_calc(p):
    if p < 16.5292:
        return _saturationTempP_calc(p)
    else:
        return 572.54459862746 + ((p - 13.91883977887) / 1.0192970039326e-3)**0.5


@vectorize([float64(float64, float64)], nopython=True, cache=True)
def _region_calc(t, p):
    if p < _borderPressureT_calc(t) and t > _borderTempP_calc(p):
        return 2 # steam
    else:
        if p == _saturationPressureT_calc(t) or t == _saturationTempP_calc(p):
            return 4 # mix
        else:
            return 1 if t < 623.15 else 3 # water or fluid


class IF97:
    def __init__(self):
        self.minP = 611.213
        self.maxP = 50
        self.minT = 273.15
        self.maxT = 1073.15
        self.t4Max = 647.096
        self.p4Max = 22.064
        self.p3Min = 16.5292
        self.t3Min = 623.15
        
        self.water = WaterRegion()
        self.steam = SteamRegion()

    def regionTP(self, t, p):
        '''Определение региона'''
        t_arr = np.asarray(t, dtype=np.float64)
        p_arr = np.asarray(p, dtype=np.float64)

        return _region_calc(t_arr, p_arr)


    def saturationPressureT(self, t):
        '''Давление насыщения через температуру'''
        t_arr = np.asarray(t, dtype=np.float64)

        return _saturationPressureT_calc(t_arr)


    def saturationTempP(self, p):
        '''Температура насыщения через давление'''
        p_arr = np.asarray(p, dtype=np.float64)

        return _saturationTempP_calc(p_arr)

    
    def borderPressureT(self, t):
        '''Давление левой границы через температуру'''
        t_arr = np.asarray(t, dtype=np.float64)

        return _borderPressureT_calc(t_arr)
        

    def borderTempP(self, p):
        '''Температура левой границы через давление'''
        p_arr = np.asarray(p, dtype=np.float64)
        
        return _borderTempP_calc(p_arr)


if97 = IF97()

if __name__ == '__main__':
    print(if97.regionTP(300, 101325))  # Скаляры
    print(if97.regionTP([300, 400], 101325))  # Массив + скаляр
    print(if97.regionTP(300, [101325, 201325]))  # Скаляр + массив
    print(if97.regionTP([300, 400], [101325, 201325]))  # Два массива
    
    print(if97.steam.enthalpy_t_p(520+273.15, [13, 14, 15]))  # Теперь работает!