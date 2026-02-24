from if97_py import mix, steam, water, fluid
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
    region_t_p = region_t_p


if97 = IF97()
