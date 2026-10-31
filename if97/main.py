'''
Единая точка входа: объект `if97`.

Всё подключается лениво. `import if97` не компилирует ни одного уравнения;
обращение к `if97.water` поднимает только область 1, к `if97.fluid` — только
область 3, к `if97.state_t_p` — все области, потому что автоматический выбор
по определению должен уметь посчитать любую.
'''

import importlib

_AREAS = {
    'water': 'Область 1: вода.',
    'steam': 'Область 2: перегретый пар.',
    'fluid': 'Область 3: сверхкритический флюид.',
    'mix': 'Область 4: пароводяная смесь.',
    'gas': 'Область 5: высокотемпературный пар.',
}

_AUTO = ('state_t_p', 'v_t_p', 'rho_t_p', 'h_t_p', 's_t_p',
         'u_t_p', 'cp_t_p', 'cv_t_p', 'w_t_p')

_BOUNDS = ('saturationPressure_t', 'saturationTemp_p', 'borderPressure_t',
           'borderTemp_p', 'b23Pressure_t', 'b23Temp_p', 'b2bcEnthalpy_p',
           'b2bcPressure_h', 'region_t_p')


class IF97:
    '''
    Свойства воды и водяного пара по IAPWS-IF97.

    `state_t_p(t, p)` сам определяет область и возвращает все восемь свойств
    за один обход таблицы коэффициентов. Обращение к области напрямую
    (`if97.water.h.t_p`) не проверяет, лежит ли точка в этой области.
    '''

    def __getattr__(self, name):
        if name in _AREAS:
            mod = importlib.import_module(f'.{name}', __package__)
        elif name in _AUTO:
            mod = getattr(importlib.import_module('.auto', __package__), name)
        elif name in _BOUNDS:
            mod = getattr(importlib.import_module('.bounds', __package__), name)
        else:
            raise AttributeError(
                f'{type(self).__name__!r} не имеет атрибута {name!r}')
        setattr(self, name, mod)   # дальше идёт обычный атрибут экземпляра
        return mod

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(_AREAS) | set(_AUTO) | set(_BOUNDS))


if97 = IF97()
