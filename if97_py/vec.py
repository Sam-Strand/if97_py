from numba import vectorize

def vec(n, nopython=True, cache=True):
    '''
    Возвращает декоратор для векторизации функции, принимающей 
    n аргументов типа float64 и возвращающей float64.
    '''
    return vectorize([f"float64({', '.join(['float64'] * n)})"], nopython=nopython, cache=cache)
