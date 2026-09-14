from ..vec import vec, Number
from ..bounds import saturationPressure_t, saturationTemp_p
from .. import steam
from .. import water


@vec(2)
def t_h(t: Number, h: Number) -> Number:
    '''
    Вычисляет степень сухости (влажность) пароводяной смеси по температуре и энтальпии.
    
    Parameters
    ----------
    t : Number
        Температура [K].
    h : Number
        Удельная энтальпия [кДж/кг].
        
    Returns
    -------
    Number
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
def p_h(p: Number, h: Number) -> Number:
    '''
    Вычисляет степень сухости (влажность) пароводяной смеси по давлению и энтальпии.
    
    Parameters
    ----------
    p : Number
        Давление [МПа].
    h : Number
        Удельная энтальпия [кДж/кг].
        
    Returns
    -------
    Number
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
def t_s(t: Number, s: Number) -> Number:
    '''
    Вычисляет степень сухости (влажность) пароводяной смеси по температуре и энтропии.
    
    Parameters
    ----------
    t : Number
        Температура [K].
    s : Number
        Удельная энтропия [кДж/(кг·K)].
        
    Returns
    -------
    Number
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
def p_s(p: Number, s: Number) -> Number:
    '''
    Вычисляет степень сухости (влажность) пароводяной смеси по давлению и энтропии.
    
    Parameters
    ----------
    p : Number
        Давление [МПа].
    s : Number
        Удельная энтропия [кДж/(кг·K)].
        
    Returns
    -------
    Number
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

