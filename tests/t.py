import numpy as np

# Эта магия находит путь к текущему файлу, берет папку выше (корень) 
# и добавляет её в список мест, где Python ищет модули потому что иначе модули питона не работают
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from if97_py import water, steam, fluid, bounds

p = (22.0, 22.064)
t = (646.89, 647.15)

v = fluid.v.t_p(t, p)

print(v[0], v[1])

'''
3.798 732 962
3.798 732 9624589563

3.701 940 010
3.701 940 009484689
'''