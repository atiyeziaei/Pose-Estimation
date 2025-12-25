# utils.py
import numpy as np

def calculate_angle(a, b, c):
    radians = np.arctan2(c.y - b.y, c.x - b.x) - \
              np.arctan2(a.y - b.y, a.x - b.x)
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return round(angle, 2)
