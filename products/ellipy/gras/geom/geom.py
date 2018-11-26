from typing import Tuple, Union
import numpy as np


def circle_part(n_points: int, return_apart: bool = False,
                angle_range_vec: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if angle_range_vec is None:
        angle_range_vec = np.array([0., 2 * np.pi])
    v_phi = np.expand_dims(np.linspace(angle_range_vec[0], angle_range_vec[1], n_points, endpoint=False), 2)
    if return_apart:
        x_vec = np.cos(v_phi)
        y_vec = np.sin(v_phi)
        return x_vec, y_vec
    else:
        return np.concatenate([np.cos(v_phi), np.sin(v_phi)], 1)


def sphere_part(n_points: int) -> np.ndarray:
    pass
