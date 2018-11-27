from typing import Tuple
import numpy as np
from scipy.special import gamma
from ellipy.elltool.conf.properties.Properties import Properties
from ellipy.gras.gen.gen import sqrt_pos
from numpy.linalg import inv, solve
from ellipy.gen.common.common import throw_error, is_numeric


def ell_volume(q_mat: np.ndarray) -> float:
    n_dims = q_mat.shape[0]
    vol_val = \
        np.pi ** (n_dims * 0.5) * np.sqrt(np.linalg.det(q_mat)) / gamma(0.5 * n_dims + 1)
    return vol_val


def inv_mat(q_mat: np.ndarray) -> np.ndarray:
    (q_mat_dim_m, q_mat_dim_n) = np.shape(q_mat)

    if q_mat_dim_m != q_mat_dim_n:
        throw_error('wrongInput', 'ELL_INV: matrix must be square.')


def quad_mat(q_mat: np.ndarray = np.array([0.]),
             x_vec: np.ndarray = np.array([0.]),
             c_vec: np.ndarray = np.array([0.]),
             mode: str = 'plain') -> float:

    if not is_numeric(q_mat):
        throw_error('wrongInput', 'q_mat must be numeric')
    if not is_numeric(x_vec):
        throw_error('wrongInput', 'x_vec must be numeric')
    if not is_numeric(c_vec):
        throw_error('wrongInput', 'c_vec must be numeric')

    if q_mat.ndim != 2:
        throw_error('wrongInput', 'q_mat must be square')
    else:
        (q_matm_elems, q_matn_elems) = q_mat.shape

    if x_vec.ndim == 1:
        (x_vecm_elems, x_vecn_elems) = (1, x_vec.shape[0])
    else:
        (x_vecm_elems, x_vecn_elems) = x_vec.shape

    if np.all(c_vec == 0):
        if x_vecm_elems == 1:
            c_vec = np.zeros(x_vecn_elems)
        else:
            c_vec = np.zeros((x_vecm_elems, x_vecn_elems))

    if c_vec.ndim == 1:
        (c_vecm_elems, c_vecn_elems) = (1, c_vec.shape[0])
    else:
        (c_vecm_elems, c_vecn_elems) = c_vec.shape

    if q_matm_elems != q_matn_elems:
        throw_error('wrongInput', 'q_mat must be square')

    if (x_vecm_elems > 1) & (x_vecn_elems > 1):
        throw_error('wrongInput', 'x_vec must be vector')
    elif x_vecm_elems > 1:
        x_vec = x_vec.T
        x_vecn_elems = x_vec.shape[1]

    if x_vecn_elems != q_matn_elems:
        throw_error('wrongInput',
                    'Dimensions of q_mat and x_vec must be coordinated')

    if (c_vecm_elems > 1) & (c_vecn_elems > 1):
        throw_error('wrongInput', 'x_vec must be vector')
    elif c_vecm_elems > 1:
        c_vec = c_vec.T
        c_vecn_elems = c_vec.shape[1]

    if c_vecn_elems != q_matn_elems:
        throw_error('wrongInput',
                    'Dimensions of q_mat and c_vec must be coordinated')

    if mode.lower() == 'plain':
        res = np.dot(x_vec - c_vec, q_mat @ (x_vec - c_vec).T)
    elif mode.lower() == 'invadv':
        res = np.dot(x_vec - c_vec, inv(q_mat) @ (x_vec - c_vec).T)
    else:
        res = (x_vec - c_vec) @ solve(q_mat, x_vec - c_vec).T

    return res

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
