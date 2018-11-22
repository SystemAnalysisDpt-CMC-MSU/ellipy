from typing import Tuple, Union
import numpy as np


def circle_part(n_points: int, return_apart: bool = False,
                angle_range_vec: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if angle_range_vec is None:
        angle_range_vec = np.array([0., 2 * np.pi])
    d_phi = (angle_range_vec[1] - angle_range_vec[0]) / n_points
    v_phi = np.r_[angle_range_vec[0] : angle_range_vec[1]: d_phi].reshape(-1,1)
    v_phi = v_phi[:-1:]
    if return_apart:
        x_mat = np.cos(v_phi)
        y_vec = np.sin(v_phi)
        return x_mat, y_vec
    x_mat = np.concatenate([np.cos(v_phi), np.sin(v_phi)], 1)
    return x_mat


def sphere_part(n_points: int) -> np.ndarray:
    pass
