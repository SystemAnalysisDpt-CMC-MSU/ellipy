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
    if ell_center_vec is None:
        m = ell_shape_mat[:, :1].shape
        ell_center_vec = np.zeros(m)
        md, nd = dirs_mat.shape
        me, ne = ell_shape_mat.shape
    if nd == 1:
        sq = sqrt_pos(np.transpose(dirs_mat).dot(ell_shape_mat.dot(dirs_mat)))
        sup_arr = np.transpose(ell_center_vec).dot(dirs_mat) + sq
        bp_mat = (ell_shape_mat.dot(dirs_mat))/sq
        bp_mat[bp_mat is None] = 0
        bp_mat = bp_mat + ell_center_vec
    else:
        temp_mat = np.transpose(sqrt_pos(np.sum(np.transpose(dirs_mat).dot(ell_shape_mat*np.transpose(dirs_mat)),
                                axis=1), abs_tol))
        sup_arr = np.transpose(ell_center_vec).dot(dirs_mat) + temp_mat
        bp_mat = ell_shape_mat.dot(dirs_mat)/np.tile(temp_mat, (me, 1))
        is_nan_bp_mat = np.isnan(bp_mat)
        is_nan_vec = np.any(is_nan_bp_mat, 0)
        if np.any(is_nan_vec):
            bp_mat[:, is_nan_vec] = 0
        bp_mat = bp_mat + np.tile(ell_center_vec, (1, nd))
    return tuple([sup_arr, bp_mat])

