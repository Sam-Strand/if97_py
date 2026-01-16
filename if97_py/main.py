import if97_py.fluid as fluid
import if97_py.mix as mix
import if97_py.steam as steam
import if97_py.water as water
from if97_py.bounds import saturationPressure_t, saturationTemp_p, borderPressure_t, borderTemp_p, region_t_p


class IF97:
    def __init__(self):
        self.fluid = fluid
        self.mix = mix
        self.steam = steam
        self.water = water

    saturationPressure_t = saturationPressure_t
    saturationTemp_p = saturationTemp_p
    borderPressure_t = borderPressure_t
    borderTemp_p = borderTemp_p
    regionTP = region_t_p


if97 = IF97()
