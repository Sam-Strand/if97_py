from if97.bounds import saturationPressure_t, saturationTemp_p, borderPressure_t, borderTemp_p, region_t_p


class IF97:
    def __init__(self):
        self._fluid = None
        self._mix = None
        self._steam = None
        self._water = None
        self._bounds = None
    
    @property
    def fluid(self):
        if self._fluid is None:
            from if97 import fluid
            self._fluid = fluid
        return self._fluid
    
    @property
    def mix(self):
        if self._mix is None:
            from if97 import mix
            self._mix = mix
        return self._mix
    
    @property
    def steam(self):
        if self._steam is None:
            from if97 import steam
            self._steam = steam
        return self._steam
    
    @property
    def water(self):
        if self._water is None:
            from if97 import water
            self._water = water
        return self._water

    saturationPressure_t = saturationPressure_t
    saturationTemp_p = saturationTemp_p
    borderPressure_t = borderPressure_t
    borderTemp_p = borderTemp_p
    region_t_p = region_t_p


if97 = IF97()
