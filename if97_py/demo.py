import time
import numpy as np
from WaterRegion import WaterRegion
from SteamRegion import SteamRegion


def quick_speed_test(n=1_000_000):
    """Быстрый тест скорости"""
    water = SteamRegion()

    # Тест на 10000 точек
    t = np.linspace(295+273.15, 305+273.15, n)
    p = np.linspace(3, 10, n)

    start = time.time()
    result = water.enthalpy_t_p(t, p)
    end = time.time()

    print(f"Скорость: {n/(end-start):.0f} расчетов/сек")
    print(f"Время: {(end-start)*1000:.1f} мс")
    print(f"Результат: {result[0]:.0f} ... {result[-1]:.0f} Дж/кг")

quick_speed_test()
