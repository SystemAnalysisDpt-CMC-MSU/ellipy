from typing import Tuple
import numpy as np


def circle_part(n_points: int, angle_range_vec: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    if angle_range_vec is None:
        angle_range_vec = np.array([0., 2 * np.pi])



def sphere_part(n_points: int) -> np.ndarray:
    pass
