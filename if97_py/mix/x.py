from if97_py.vec import vec
from if97_py.bounds import saturationPressure_t, saturationTemp_p
import if97_py.steam as steam
import if97_py.water as water


@vec(2)
def t_h(t, h):
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
def p_h(p, h):
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
def t_s(t, s):
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
def p_s(p, s):
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

