import numpy as np


def check(diff):
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    return f'{RED if np.any(diff != 0) else GREEN}{diff}{RESET}'


'''
Регион 1

Тестовые данные
'''
t = (300, 300, 500)
p = (3, 80, 3)

v = (0.0010021516796866943, 0.0009711808940216298, 0.001202418003378339)
h = (115.33127302143839, 184.14282773425435, 975.542239097225)
s = (0.3922947924026242, 0.3685638523984806, 2.58041912005181)
w = (1507.739209669031, 1634.6905431116586, 1240.713373101725)

import if97_py.water as water
print('volume_t_p', check(water.volume_t_p(t, p) - v))
print('enthalpy_t_p', check(water.enthalpy_t_p(t, p) - h))
print('entropy_t_p', check(water.entropy_t_p(t, p) - s))
print('soundSpeed_t_p', check(water.soundSpeed_t_p(t, p) - w))

'''
Регион 2

Тестовые данные
'''

t = (300, 700, 700)
p = (0.0035, 0.0035, 30)

v = (39.491386637762986, 92.30158981741968, 0.005429466194617729)
h = (2549.9114508400203, 3335.683753731224, 2631.4947448448083)
s = (8.522389667335792, 10.174999578595989, 5.175402982299071)
w = (427.9201722631048, 644.2890675665433, 480.38652316973463)

import if97_py.steam as steam
print('volume_t_p', check(steam.volume_t_p(t, p) - v))
print('enthalpy_t_p', check(steam.enthalpy_t_p(t, p) - h))
print('entropy_t_p', check(steam.entropy_t_p(t, p) - s))
print('soundSpeed_t_p', check(steam.soundSpeed_t_p(t, p) - w))
