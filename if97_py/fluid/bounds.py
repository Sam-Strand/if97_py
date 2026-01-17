from math import exp, log
from if97_py.vec import vec

@vec(1)
def _t_cd(p):
    '''Boundary between 3c-3d, T = f(P)'''
    return (0.585276966696349e3 + 
           0.278233532206915e1 * p + 
            -0.127283549295878e-1 * p**2 + 
            0.159090746562729e-3 * p**3)

@vec(1)
def _t_gh(p):
    '''Boundary between 3g-3h, T = f(P)'''
    return (-0.249284240900418e5 + 
            0.428143584791546e4 * p + 
            -0.269029173140130e3 * p**2 + 
            0.751608051114157e1 * p**3 + 
            -0.787105249910383e-1 * p**4)

@vec(1)
def _t_ij(p):
    '''Boundary between 3i-3j, T = f(P)'''
    return (0.584814781649163e3 + 
            -0.616179320924617 * p + 
            0.260763050899562 * p**2 + 
            -0.587071076864459e-2 * p**3 + 
            0.515308185433082e-4 * p**4)

@vec(1)
def _t_jk(p):
    '''Boundary between 3j-3k, T = f(P)'''
    return (0.617229772068439e3 + 
            -0.770600270141675e1 * p + 
            0.697072596851896 * p**2 + 
            -0.157391839848015e-1 * p**3 + 
           0.137897492684194e-3 * p**4)

@vec(1)
def _t_mn(p):
    '''Boundary between 3m-3n, T = f(P)'''
    return (0.535339483742384e3 + 
            0.761978122720128e1 * p + 
            -0.158365725441648 * p**2 + 
            0.192871054508108e-2 * p**3)

@vec(1)
def _t_qu(p):
    '''Boundary between 3q-3u, T = f(P)'''
    return (0.565603648239126e3 + 
            0.529062258221222e1 * p + 
            -0.102020639611016 * p**2 + 
            0.122240301070145e-2 * p**3)


@vec(1)
def _t_rx(p):
    '''Boundary between 3r-3x, T = f(P)'''
    return (0.584561202520006e3 + 
            -0.102961025163669e1 * p + 
            0.243293362700452 * p**2 + 
            -0.294905044740799e-2 * p**3)


@vec(1)
def _t_uv(p):
    '''Boundary between 3u-3v, T = f(P)'''
    return (0.528199646263062e3 + 
            0.890579602135307e1 * p + 
            -0.222814134903755 * p**2 + 
            0.286791682263697e-2 * p**3)


@vec(1)
def _tab_P(p):
    '''
    Define the boundary between Region 3a-3b, T=f(P)
    
    Parameters
    ----------
    p : float
        Pressure, [MPa]
    
    Returns
    -------
    t : float
        Temperature, [K]
    '''
    log_p = log(p)
    log_p2 = log_p * log_p  # log_p ** 2
    inv_log_p = 1.0 / log_p
    inv_log_p2 = inv_log_p * inv_log_p  # log_p ** -2
    
    return (
        0.154793642129415e4 +
        -0.187661219490113e3 * log_p +
        0.213144632222113e2 * log_p2 +
        -0.191887498864292e4 * inv_log_p +
        0.918419702359447e3 * inv_log_p2
    )


@vec(1)
def _twx_P(p):
    '''
    Define the boundary between Region 3w-3x, T=f(P)

    Parameters
    ----------
    p : float
        Pressure, [MPa]

    Returns
    -------
    t : float
        Temperature, [K]
    '''
    log_p = log(p)
    log_p2 = log_p * log_p  # log_p ** 2
    inv_log_p = 1.0 / log_p
    inv_log_p2 = inv_log_p * inv_log_p  # log_p ** -2
    
    return (
        0.728052609145380e1 +
        0.973505869861952e2 * log_p +
        0.147370491183191e2 * log_p2 +
        0.329196213998375e3 * inv_log_p +
        0.873371668682417e3 * inv_log_p2
    )


@vec(1)
def _tef_P(p):
    '''Define the boundary between Region 3e-3f, T=f(P)

    Parameters
    ----------
    p : float
        Pressure, [MPa]

    Returns
    -------
    t : float
        Temperature, [K]
    '''
    return 3.727888004 * (p - 22.064) + 647.096


@vec(1)
def _top_P(p):
    '''
    Define the boundary between Region 3o-3p, T=f(P)

    Parameters
    ----------
    p : float
        Pressure, [MPa]

    Returns
    -------
    t : float
        Temperature, [K]
    '''
    log_p = log(p)
    log_p2 = log_p * log_p  # log_p ** 2
    inv_log_p = 1.0 / log_p
    inv_log_p2 = inv_log_p * inv_log_p  # log_p ** -2
    
    return (
        0.969461372400213e3 +
        -0.332500170441278e3 * log_p +
        0.642859598466067e2 * log_p2 +
        0.773845935768222e3 * inv_log_p +
        -0.152313732937084e4 * inv_log_p2
    )
