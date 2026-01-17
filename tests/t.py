import numpy as np

# Эта магия находит путь к текущему файлу, берет папку выше (корень) 
# и добавляет её в список мест, где Python ищет модули потому что иначе модули питона не работают
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from if97_py import water, steam, fluid, bounds




print(bounds.saturationPressure_t(643.15)) # 21.04336731897525
print(bounds.saturationPressure_t(623.15)) # 16.52916425260448


