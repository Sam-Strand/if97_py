'''
Публичный слой: один и тот же njit-код обслуживает и скаляр, и массив.

Раньше свойства были ufunc'ами (`@vectorize`). Вызов ufunc'а со скалярами
стоит около 1.8 мкс на разрешение типов, упаковку скаляров в массивы и
аллокацию выхода — при том, что сама математика занимает доли микросекунды.
Здесь вместо этого проверка типа в Python и прямой вызов: скаляр идёт в
njit-функцию, массив — в njit-цикл, который пересекает границу Python ровно
один раз на весь массив.

Проверку «скаляр ли это» приходится писать аккуратно. Очевидное
`np.ndim(a) == 0 and np.ndim(b) == 0` стоит 1570 нс на пару аргументов и
полностью съедает выигрыш от отказа от ufunc'а. `a.__class__ is float`
стоит 28 нс, то есть в полсотни раз меньше, и покрывает самый частый случай;
всё остальное уходит на медленную ветку, где лишние наносекунды уже не видны.

Таблицы областей передаются только пакетным циклам. Распаковка массива в
сигнатуре njit-функции стоит 85 нс на массив при вызове из Python: циклу это
окупается один раз на весь массив, а точечному вызову — нет. Поэтому
скалярные ядра замкнуты на свои таблицы и принимают только (t, p), а `extra`
уходит исключительно в `batch`.

Побочный эффект shared-power: посчитать в точке одно свойство стоит столько
же, сколько все восемь. Поэтому `state.t_p` — основной вход, а отдельные
`h.t_p`, `cp.t_p` и прочие берут из него нужное поле. На массивах
`state.t_p` заметно выгоднее: он возвращает все свойства за ту же работу,
тогда как восемь отдельных вызовов восемь раз повторят один и тот же обход.
'''

from collections import namedtuple

import numpy as np

#: Полное состояние однофазной области.
State = namedtuple('State', 'v rho h s u cp cv w')

#: Состояние двухфазной смеси: cp, cv и скорость звука для неё не определены.
MixState = namedtuple('MixState', 'v rho h s u')

NPROP = len(State._fields)
IDX = {name: i for i, name in enumerate(State._fields)}

#: Что считаем скаляром на медленной ветке: int, numpy-скаляры и прочее.
_NUM = (float, int, np.floating, np.integer)


def _pair(a, b):
    '''Приводит пару аргументов к одномерным непрерывным массивам float64.'''
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    shape = a.shape
    return (np.ascontiguousarray(a).ravel(),
            np.ascontiguousarray(b).ravel(), shape)


def make_state(scalar, batch, extra=(), cls=State):
    '''Полное состояние: namedtuple величин (скаляров или массивов).'''
    n = len(cls._fields)

    if extra:
        def f(a, b):
            if a.__class__ is float and b.__class__ is float:
                return cls(*scalar(a, b))
            if isinstance(a, _NUM) and isinstance(b, _NUM):
                return cls(*scalar(float(a), float(b)))
            x, y, shape = _pair(a, b)
            out = np.empty((n, x.size), dtype=np.float64)
            batch(x, y, out, *extra)
            if shape == ():
                return cls(*(float(out[i, 0]) for i in range(n)))
            return cls(*(out[i].reshape(shape) for i in range(n)))
    else:
        def f(a, b):
            if a.__class__ is float and b.__class__ is float:
                return cls(*scalar(a, b))
            if isinstance(a, _NUM) and isinstance(b, _NUM):
                return cls(*scalar(float(a), float(b)))
            x, y, shape = _pair(a, b)
            out = np.empty((n, x.size), dtype=np.float64)
            batch(x, y, out)
            if shape == ():
                return cls(*(float(out[i, 0]) for i in range(n)))
            return cls(*(out[i].reshape(shape) for i in range(n)))
    return f


def make_prop(scalar, batch, name, extra=(), cls=State):
    '''Одно свойство из полного состояния.'''
    n = len(cls._fields)
    i = cls._fields.index(name)

    if extra:
        def f(a, b):
            if a.__class__ is float and b.__class__ is float:
                return scalar(a, b)[i]
            if isinstance(a, _NUM) and isinstance(b, _NUM):
                return scalar(float(a), float(b))[i]
            x, y, shape = _pair(a, b)
            out = np.empty((n, x.size), dtype=np.float64)
            batch(x, y, out, *extra)
            return float(out[i, 0]) if shape == () else out[i].reshape(shape)
    else:
        def f(a, b):
            if a.__class__ is float and b.__class__ is float:
                return scalar(a, b)[i]
            if isinstance(a, _NUM) and isinstance(b, _NUM):
                return scalar(float(a), float(b))[i]
            x, y, shape = _pair(a, b)
            out = np.empty((n, x.size), dtype=np.float64)
            batch(x, y, out)
            return float(out[i, 0]) if shape == () else out[i].reshape(shape)
    return f


def make1(scalar, batch):
    '''Функция одного аргумента: скаляр или массив.'''
    def f(a):
        if a.__class__ is float:
            return scalar(a)
        if isinstance(a, _NUM):
            return scalar(float(a))
        x = np.asarray(a, dtype=np.float64)
        shape = x.shape
        x = np.ascontiguousarray(x).ravel()
        out = np.empty(x.size, dtype=np.float64)
        batch(x, out)
        return float(out[0]) if shape == () else out.reshape(shape)
    return f


def make2(scalar, batch, extra=(), dtype=np.float64):
    '''Функция двух аргументов, возвращающая одно число.'''
    if extra:
        def f(a, b):
            if a.__class__ is float and b.__class__ is float:
                return scalar(a, b)
            if isinstance(a, _NUM) and isinstance(b, _NUM):
                return scalar(float(a), float(b))
            x, y, shape = _pair(a, b)
            out = np.empty(x.size, dtype=dtype)
            batch(x, y, out, *extra)
            return out[0].item() if shape == () else out.reshape(shape)
    else:
        def f(a, b):
            if a.__class__ is float and b.__class__ is float:
                return scalar(a, b)
            if isinstance(a, _NUM) and isinstance(b, _NUM):
                return scalar(float(a), float(b))
            x, y, shape = _pair(a, b)
            out = np.empty(x.size, dtype=dtype)
            batch(x, y, out)
            return out[0].item() if shape == () else out.reshape(shape)
    return f
