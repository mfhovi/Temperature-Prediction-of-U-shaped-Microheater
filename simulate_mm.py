import numpy as np

def simulate_mm(params):
    W_mm, T_hot = params
    width_mm = 10.0
    height_mm = 10.0
    total_length_mm = 4.0
    d_mm = 1.0
    max_iter = 10000
    tol = 1e-3

    fine_res_mm = 0.01
    coarse_res_mm = 0.1

    H_mm = (total_length_mm - W_mm) / 2
    hot_zone_y_min = (height_mm - H_mm) / 2
    hot_zone_y_max = (height_mm + H_mm) / 2
    Tm_y = hot_zone_y_min - d_mm

    x_fine = np.arange((width_mm - W_mm) / 2, (width_mm + W_mm) / 2 + fine_res_mm, fine_res_mm)
    x_coarse_left = np.arange(0, (width_mm - W_mm) / 2, coarse_res_mm)
    x_coarse_right = np.arange((width_mm + W_mm) / 2 + coarse_res_mm, width_mm + coarse_res_mm, coarse_res_mm)
    x = np.unique(np.concatenate((x_coarse_left, x_fine, x_coarse_right)))

    y_fine = np.arange(Tm_y - 0.1, hot_zone_y_max + 0.5 + fine_res_mm, fine_res_mm)
    y_coarse_bottom = np.arange(0, hot_zone_y_min - 0.5, coarse_res_mm)
    y_coarse_top = np.arange(hot_zone_y_max + 0.5 + coarse_res_mm, height_mm + coarse_res_mm, coarse_res_mm)
    y = np.unique(np.concatenate((y_coarse_bottom, y_fine, y_coarse_top)))

    nx, ny = len(x), len(y)
    T = np.ones((nx, ny)) * 25

    ix_left = np.searchsorted(x, (width_mm - W_mm) / 2)
    ix_right = np.searchsorted(x, (width_mm + W_mm) / 2)
    iy_bottom = np.searchsorted(y, hot_zone_y_min)
    iy_top = np.searchsorted(y, hot_zone_y_max)

    T[ix_left, iy_bottom:iy_top + 1] = T_hot
    T[ix_right, iy_bottom:iy_top + 1] = T_hot
    T[ix_left:ix_right + 1, iy_bottom] = T_hot

    for i in range(max_iter):
        T_old = T.copy()
        T[1:-1, 1:-1] = 0.25 * (T_old[2:, 1:-1] + T_old[:-2, 1:-1] + T_old[1:-1, 2:] + T_old[1:-1, :-2])
        T[ix_left, iy_bottom:iy_top + 1] = T_hot
        T[ix_right, iy_bottom:iy_top + 1] = T_hot
        T[ix_left:ix_right + 1, iy_bottom] = T_hot
        T[0, :] = T[1, :]
        T[-1, :] = T[-2, :]
        T[:, 0] = T[:, 1]
        T[:, -1] = T[:, -2]
        if np.max(np.abs(T - T_old)) < tol:
            break

    Tc_region = T[ix_left:ix_right + 1, iy_bottom:iy_top + 1]
    Tc_avg = np.mean(Tc_region)
    Tc_std = np.std(Tc_region)

    iy_d = np.searchsorted(y, Tm_y)
    ix_center = np.searchsorted(x, width_mm / 2)
    ix_range = slice(max(0, ix_center - 1), min(nx, ix_center + 2))
    iy_range = slice(max(0, iy_d - 1), min(ny, iy_d + 2))
    Tm = np.mean(T[ix_range, iy_range])

    ix_right_shift = np.searchsorted(x, x[ix_center] + 2.0)
    ix2_range = slice(max(0, ix_right_shift - 1), min(nx, ix_right_shift + 2))
    Tm2 = np.mean(T[ix2_range, iy_range])

    return W_mm, T_hot, Tc_avg, Tc_std, Tm, Tm2
