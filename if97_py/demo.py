import time
import numpy as np
from SteamRegion import SteamRegion as Region
# from WaterRegion import WaterRegion as Region


def simple_speed_test(n=1_000_000, runs=10):
    """Простой тест скорости"""

    region = Region()

    t = np.linspace(295 + 273.15, 305 + 273.15, n)
    p = np.linspace(3, 10, n)

    print(f"Запуск {runs} прогонов...\n")

    speeds = []

    for i in range(runs):
        start = time.perf_counter()
        result = region.enthalpy_t_p(t, p)
        end = time.perf_counter()

        speed = n / (end - start)
        speeds.append(speed)

        print(f"Прогон {i+1:2d}: {speed:,.0f} расчетов/сек", result[0])

    avg_speed = np.mean(speeds)
    print(f"\nСредняя скорость: {avg_speed:,.0f} расчетов/сек")
    print(f"Отклонение: ±{np.std(speeds) / avg_speed * 100:.1f}%")

    return avg_speed


if __name__ == "__main__":
    print("=" * 50)
    print("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 50)

    result = simple_speed_test(n=10_000_000, runs=10)

    print(f"\nИтог: ~{result:,.0f} расчетов энтальпии в секунду")
    print(f"На одном ядре CPU")
