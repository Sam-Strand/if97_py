'''
Разделяемая механика обратных уравнений.

Обратные уравнения — простые суммы n·x^I·y^J: ни производных, ни shared-power,
нужна одна величина, а не шесть. Остаётся только лестница степеней — та же,
что в прямых уравнениях. Степени набираются один раз умножением, и внутренний
цикл сводится к двум выборкам из массива и умножению, без единого вызова pow.

Показатели в таблицах стандарта доходят до 44 по одной оси и до -27 по другой,
так что наивный `x ** I` в цикле — это сотня обращений к pow на точку.
'''

from numba import njit

from ._jit import JIT


@njit(**JIT)
def ladder(pw, v, lo, hi):
    '''Заполняет pw[k - lo] = v ** k для k от lo до hi; lo может быть < 0.'''
    pw[-lo] = 1.0
    for k in range(1, hi + 1):
        pw[k - lo] = pw[k - lo - 1] * v
    if lo < 0:
        inv = 1.0 / v
        for k in range(-1, lo - 1, -1):
            pw[k - lo] = pw[k - lo + 1] * inv


@njit(**JIT)
def summed(x, y, I, J, N, xlo, xhi, ylo, yhi, pw_x, pw_y):
    '''Сумма n_k · x^(I_k) · y^(J_k) по двум лестницам степеней.'''
    ladder(pw_x, x, xlo, xhi)
    ladder(pw_y, y, ylo, yhi)
    acc = 0.0
    for k in range(N.shape[0]):
        acc += N[k] * pw_x[I[k] - xlo] * pw_y[J[k] - ylo]
    return acc
