from typing import Tuple
import numpy as np
from ellipy.elltool.conf.properties.Properties import Properties
from ellipy.gras.gen.gen import sqrt_pos


def ell_volume(q_mat: np.ndarray) -> float:
    pass


def inv_mat(q_mat: np.ndarray) -> np.ndarray:
    pass


def quad_mat(q_mat: np.ndarray, x_vec: np.ndarray, cVec: np.ndarray = np.array([0.]), mode: str = 'plain') -> float:
    pass


def rho_mat(ell_shape_mat: np.ndarray, dirs_mat: np.ndarray,
            abs_tol: float = None, ell_center_vec:  np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    if abs_tol is None:
        abs_tol = Properties.get_abs_tol()
    m = ell_shape_mat.shape[0]
    if ell_center_vec is None:
        ell_center_vec = np.zeros((m, 1))
    nd = dirs_mat.shape[1]
    if nd == 1:
        sq = sqrt_pos(dirs_mat.T @ ell_shape_mat @ dirs_mat)
        sup_arr = ell_center_vec.T @ dirs_mat + sq
        bp_mat = (ell_shape_mat @ dirs_mat) / sq
        if np.any(np.isnan(bp_mat)):
            bp_mat[:] = 0.
        bp_mat = bp_mat + ell_center_vec
    else:
        temp_mat = sqrt_pos(np.sum(dirs_mat.T @ ell_shape_mat * dirs_mat.T,
                                axis=1), abs_tol).T
        sup_arr = ell_center_vec.T @ dirs_mat + temp_mat
        bp_mat = ell_shape_mat @ dirs_mat / np.tile(temp_mat, (m, 1))
        is_nan_bp_mat = np.isnan(bp_mat)
        is_nan_vec = np.any(is_nan_bp_mat, 0)
        if np.any(is_nan_vec):
            bp_mat[:, is_nan_vec] = 0.
        bp_mat = bp_mat + np.tile(ell_center_vec, (1, nd))
    return sup_arr, bp_mat

