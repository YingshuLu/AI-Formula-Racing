import cv2
import numpy as np

def flatten_red(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter] = 255, 255
        b[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        #flattened = cv2.merge((r, g, b))
        return b

def wall_angle(left_wall_distance, right_wall_distance, MAX_STEERING=6.5):
        count_rate = (right_wall_distance / left_wall_distance) if left_wall_distance > 0 else 0
        count_rate = max(min(count_rate * 40 , MAX_STEERING), 0)
        steering_angle = -count_rate if right_wall_distance > left_wall_distance else count_rate
        return steering_angle
