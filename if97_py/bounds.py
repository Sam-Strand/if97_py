from math import sqrt
from if97_py.vec import vec


@vec(1)
def saturationPressure_t(t):
    K1 = t - 0.23855557567849 / (t - 650.17534844798)
    K2 = -17.073846940092 * K1**2 + 12020.82470247 * K1 - 3232555.0322333
    K3 = 14.91510861353 * K1**2 - 4823.2657361591 * K1 + 405113.40542057
    
    discriminant = K2**2 - 4 * (K1**2 + 1167.0521452767 * K1 - 724213.16703206) * K3
    denominator = -K2 + sqrt(discriminant)
    
    return (2 * K3 / denominator)**4


@vec(1)
def saturationTemp_p(p):
    k1 = 650.17534844798
    p_quarter = p**0.25
    
    K1 = 1167.0521452767 * p_quarter**2 + 12020.82470247 * p_quarter - 4823.2657361591
    K2 = -724213.16703206 * p_quarter**2 - 3232555.0322333 * p_quarter + 405113.40542057
    discriminant = K1**2 - 4 * (p_quarter**2 - 17.073846940092 * p_quarter + 14.91510861353) * K2
    K3 = 2 * K2 / (-K1 - sqrt(discriminant))
    
    temp_k = (k1 + K3 - sqrt((k1 + K3)**2 - 4 * (k1 * K3 - 0.23855557567849))) / 2
    return temp_k


@vec(1)
def borderPressure_t(t):
    if t <= 623.15:
        return saturationPressure_t(t)
    else:
        return (348.05185628969 - 1.1671859879975 * t + 1.0192970039326e-3 * t**2)
    

@vec(1)
def borderTemp_p(p):
    if p < 16.5292:
        return saturationTemp_p(p)
    else:
        return 572.54459862746 + ((p - 13.91883977887) / 1.0192970039326e-3)**0.5


@vec(2)
def regionTP(t, p):
    if p < borderPressure_t(t) and t > borderTemp_p(p):
        return 2 # steam
    else:
        if p == saturationPressure_t(t) or t == saturationTemp_p(p):
            return 4 # mix
        else:
            return 1 if t < 623.15 else 3 # water or fluid