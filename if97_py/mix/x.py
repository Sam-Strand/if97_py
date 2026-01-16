from if97_py.vec import vec
from if97_py.bounds import saturationPressure_t, saturationTemp_p
import if97_py.steam as steam
import if97_py.water as water
from if97_py.types import ArrayLike, FloatArray


@vec(2)
def t_h(t: ArrayLike, h: ArrayLike) -> FloatArray:
    '''
    Вычисляет степень сухости (влажность) пароводяной смеси по температуре и энтальпии.
    
    Parameters
    ----------
    t : ArrayLike
        Температура [K].
    h : ArrayLike
        Удельная энтальпия [кДж/кг].
        
    Returns
    -------
    FloatArray
        Степень сухости x [доля] в диапазоне [0, 1]:
        - 0: насыщенная вода (h < h')
        - 1: сухой насыщенный пар (h > h'')
        - 0 < x < 1: пароводяная смесь
    '''
    p = saturationPressure_t(t)
    if p:
        h1 = water.h.t_p(t, p)
        if h1 < h:
            h2 = steam.h.t_p(t, p)
            if h2 > h:
                return (h - h1) / (h2 - h1)
            else:
                return 1
        else:
            return 0
    else:
        return 1


@vec(2)
def p_h(p: ArrayLike, h: ArrayLike) -> FloatArray:
    '''
    Вычисляет степень сухости (влажность) пароводяной смеси по давлению и энтальпии.
    
    Parameters
    ----------
    p : ArrayLike
        Давление [МПа].
    h : ArrayLike
        Удельная энтальпия [кДж/кг].
        
    Returns
    -------
    FloatArray
        Степень сухости x [доля] в диапазоне [0, 1]:
        - 0: насыщенная вода (h < h')
        - 1: сухой насыщенный пар (h > h'')
        - 0 < x < 1: пароводяная смесь
    '''
    t = saturationTemp_p(p)
    if t:
        h1 = water.h.t_p(t, p)
        if h1 < h:
            h2 = steam.h.t_p(t, p)
            if h2 > h:
                return (h - h1) / (h2 - h1)
            else:
                return 1
        else:
            return 0
    else:
        return 1

    
@vec(2)
def t_s(t: ArrayLike, s: ArrayLike) -> FloatArray:
    '''
    Вычисляет степень сухости (влажность) пароводяной смеси по температуре и энтропии.
    
    Parameters
    ----------
    t : ArrayLike
        Температура [K].
    s : ArrayLike
        Удельная энтропия [кДж/(кг·K)].
        
    Returns
    -------
    FloatArray
        Степень сухости x [доля] в диапазоне [0, 1]:
        - 0: насыщенная вода (s < s')
        - 1: сухой насыщенный пар (s > s'')
        - 0 < x < 1: пароводяная смесь
    '''
    p = saturationPressure_t(t)
    if p:
        s1 = water.s.t_p(t, p)
        if s1 < s:
            s2 = steam.s.t_p(t, p)
            if s2 > s:
                return (s - s1) / (s2 - s1)
            else:
                return 1
        else:
            return 0
    else:
        return 1


@vec(2)
def p_s(p: ArrayLike, s: ArrayLike) -> FloatArray:
    '''
    Вычисляет степень сухости (влажность) пароводяной смеси по давлению и энтропии.
    
    Parameters
    ----------
    p : ArrayLike
        Давление [МПа].
    s : ArrayLike
        Удельная энтропия [кДж/(кг·K)].
        
    Returns
    -------
    FloatArray
        Степень сухости x [доля] в диапазоне [0, 1]:
        - 0: насыщенная вода (s < s')
        - 1: сухой насыщенный пар (s > s'')
        - 0 < x < 1: пароводяная смесь
    '''
    t = saturationTemp_p(p)
    if t:
        s1 = water.s.t_p(t, p)
        if s1 < s:
            s2 = steam.s.t_p(t, p)
            if s2 > s:
                return (s - s1) / (s2 - s1)
            else:
                return 1
        else:
            return 0
    else:
        return 1

