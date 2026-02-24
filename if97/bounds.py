from math import sqrt
from if97.vec import vec
from if97.consts import p3Min, t3Min, minP, maxP, minT, maxT, p4Max, t4Max
from numpy import nan
from if97.types import ArrayLike, FloatArray


@vec(1)
def saturationPressure_t(t: ArrayLike) -> FloatArray:
    '''
    Вычисляет давление насыщения по температуре.
    Соответствует линии [4] (пароводяной смеси) в PT-диаграмме.
    
    Parameters
    ----------
    t : ArrayLike
        Температура [K].
        
    Returns
    -------
    FloatArray
        Давление насыщения [МПа]. NaN для значений вне диапазона.
        
    Notes
    -----
    Диапазон: minT ≤ t ≤ t4Max
    '''
    if t > t4Max or t < minT: # Выход за границы
        return nan
    K1 = t - 0.23855557567849 / (t - 650.17534844798)
    K2 = -17.073846940092 * K1**2 + 12020.82470247 * K1 - 3232555.0322333
    K3 = 14.91510861353 * K1**2 - 4823.2657361591 * K1 + 405113.40542057
    return (2 * K3 / (-K2 + sqrt(K2**2 - 4 * (K1**2 + 1167.0521452767 * K1 - 724213.16703206) * K3)))**4


@vec(1)
def saturationTemp_p(p: ArrayLike) -> FloatArray:
    '''
    Вычисляет температуру насыщения по давлению.
    Соответствует линии [4] (пароводяной смеси) в PT-диаграмме.
    
    Parameters
    ----------
    p : ArrayLike
        Давление [МПа].
        
    Returns
    -------
    FloatArray
        Температура насыщения [K]. NaN для значений вне диапазона.
    '''
    if p > p4Max or p < minP: # Выход за границы
        return nan
    k1 = 650.17534844798
    p_quarter = p**0.25
    K1 = 1167.0521452767 * p_quarter**2 + 12020.82470247 * p_quarter - 4823.2657361591
    K2 = -724213.16703206 * p_quarter**2 - 3232555.0322333 * p_quarter + 405113.40542057
    K3 = 2 * K2 / (-K1 - sqrt(K1**2 - 4 * (p_quarter**2 - 17.073846940092 * p_quarter + 14.91510861353) * K2))
    return (k1 + K3 - sqrt((k1 + K3)**2 - 4 * (k1 * K3 - 0.23855557567849))) / 2


@vec(1)
def borderPressure_t(t: ArrayLike) -> FloatArray:
    '''
    Давление на левой границе области 2 (перегретый пар).
    
    При t ≤ t3Min: граница с областью 4 (насыщение).
    При t > t3Min: граница с областью 3 (сверхкритическая).

    Parameters
    ----------
    t : ArrayLike
        Температура [K].
        
    Returns
    -------
    FloatArray
        Давление на границе [МПа].
    '''
    if t <= t3Min: # пересечение с 4
        return saturationPressure_t(t)
    else: # пересечение с 3
        return 348.05185628969 - 1.1671859879975 * t + 1.0192970039326e-3 * t**2
    

@vec(1)
def borderTemp_p(p: ArrayLike) -> FloatArray:
    '''
    Температура на левой границе области 2 (перегретый пар).
    
    При p < p3Min: граница с областью 4 (насыщение).
    При p ≥ p3Min: граница с областью 3 (сверхкритическая).

    Parameters
    ----------
    p : ArrayLike
        Давление [МПа].
        
    Returns
    -------
    FloatArray
        Температура на границе [K].
    '''
    if p < p3Min: # пересечение с 4
        return saturationTemp_p(p)
    else: # пересечение с 3
        return 572.54459862746 + sqrt((p - 13.91883977887) / 1.0192970039326e-3)


@vec(2)
def region_t_p(t: ArrayLike, p: ArrayLike) -> FloatArray:
    '''
    Определяет термодинамическую область по температуре и давлению.
    
    Parameters
    ----------
    t : ArrayLike
        Температура [K].
    p : ArrayLike
        Давление [МПа].
        
    Returns
    -------
    FloatArray
        Код области: 
        1 - вода (water)
        2 - перегретый пар (steam) 
        3 - сверхкритическая жидкость (fluid)
        4 - пароводяная смесь (mix)
        nan - ошибка (вне диапазона)
        
    Notes
    -----
    Используется точное сравнение (==) для определения границы насыщения.
    '''
    if p < minP or p > maxP or t < minT or t > maxT:
        return nan
    if p < borderPressure_t(t) and t > borderTemp_p(p):
        return 2 # steam
    else:
        if p == saturationPressure_t(t) or t == saturationTemp_p(p):
            return 4 # mix
        else:
            return 1 if t < t3Min else 3 # water or fluid
