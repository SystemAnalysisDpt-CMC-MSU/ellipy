from typing import Tuple
import numpy as np


def ell_volume(q_mat: np.ndarray) -> float:
    pass


def inv_mat(q_mat: np.ndarray) -> np.ndarray:
    pass


def quad_mat(q_mat: np.ndarray, x_vec: np.ndarray, cVec: np.ndarray = np.array([0.]), mode: str = 'plain') -> float:
    pass


def rho_mat(ell_shape_mat: np.ndarray, dirs_mat: np.ndarray,
            abs_tol: float = None, ell_center_vec:  np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    pass
