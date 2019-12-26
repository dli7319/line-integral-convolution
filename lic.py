import numpy as np
from numba import njit


def get_noise(vectors):
    return np.random.rand(*(vectors.shape[0:2]))


@njit
def lic_flow(vectors, t=0, len_pix=5, noise=None):
    vectors = np.asarray(vectors)
    m, n, two = vectors.shape
    if two != 2:
        raise ValueError

    if noise is None:
        noise = np.random.rand(*(vectors.shape[0:2]))

    result = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            y = i
            x = j
            forward_sum = 0
            forward_total = 0
            # Advect forwards
            for k in range(len_pix):
                dx = vectors[int(y), int(x), 0]
                dy = vectors[int(y), int(x), 1]
                dt_x = dt_y = 0
                if dy > 0:
                    dt_y = ((np.floor(y) + 1) - y) / dy
                elif dy < 0:
                    dt_y = (y - (np.ceil(y) - 1)) / -dy
                if dx > 0:
                    dt_x = ((np.floor(x) + 1) - x) / dx
                elif dx < 0:
                    dt_x = (x - (np.ceil(x) - 1)) / -dx
                if dx == 0 and dy == 0:
                    dt = 0
                else:
                    dt = min(dt_x, dt_y)
                x = min(max(x + dx * dt, 0), n - 1)
                y = min(max(y + dy * dt, 0), m - 1)
                weight = pow(np.cos(t + 0.46 * k), 2)
                forward_sum += noise[int(y), int(x)] * weight
                forward_total += weight
            y = i
            x = j
            backward_sum = 0
            backward_total = 0
            # Advect backwards
            for k in range(1, len_pix):
                dx = vectors[int(y), int(x), 0]
                dy = vectors[int(y), int(x), 1]
                dy *= -1
                dx *= -1
                dt_x = dt_y = 0
                if dy > 0:
                    dt_y = ((np.floor(y) + 1) - y) / dy
                elif dy < 0:
                    dt_y = (y - (np.ceil(y) - 1)) / -dy
                if dx > 0:
                    dt_x = ((np.floor(x) + 1) - x) / dx
                elif dx < 0:
                    dt_x = (x - (np.ceil(x) - 1)) / -dx
                if dx == 0 and dy == 0:
                    dt = 0
                else:
                    dt = min(dt_x, dt_y)
                x = min(max(x + dx * dt, 0), n - 1)
                y = min(max(y + dy * dt, 0), m - 1)
                weight = pow(np.cos(t - 0.46 * k), 2)
                backward_sum += noise[int(y), int(x)] * weight
                backward_total += weight
            result[i, j] = (forward_sum + backward_sum) / (forward_total + backward_total)
    return result
