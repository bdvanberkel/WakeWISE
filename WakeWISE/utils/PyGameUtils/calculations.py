import math as m
import numpy as np


def pywake_to_math_rad(dir):

    dir %= 360

    # WD is defined as zero being up, going clockwise
    # rad is defined as zero being right, going anticlockwise

    # return -np.deg2rad(dir - 90)
    return -np.deg2rad(dir - 90) % (2 * np.pi)

def value_map(input_min, input_max, map_min, map_max, x):
    x = min(max(x, input_min), input_max)
    return ((x - input_min) / (input_max - input_min)) * (map_max - map_min) + map_min


if __name__ == "__main__":

    print(pywake_to_math_rad(0))
    print(pywake_to_math_rad(90))
    print(pywake_to_math_rad(180))
    print(pywake_to_math_rad(270))
    print(pywake_to_math_rad(225))

    assert pywake_to_math_rad(0) == m.pi / 2
    assert pywake_to_math_rad(90) == 0
    assert pywake_to_math_rad(180) == 1.5 * m.pi
    assert pywake_to_math_rad(270) == m.pi