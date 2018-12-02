from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.gras.la.la import *
from ellipy.gen.common.common import throw_error, is_numeric
import numpy as np


def ell_unitball(n: int):
    return Ellipsoid(np.eye(n))


def ell_sim_diag(a_mat: np.ndarray, b_mat: np.ndarray, abs_tol: float) -> np.ndarray:
    if not (is_numeric(a_mat) and is_numeric(b_mat)):
        throw_error('wrongInput', 'both arguments must be numeric matrices')
    if not is_mat_pos_def(a_mat, abs_tol):
        throw_error('wrongInput:a_mat', 'first argument must be a symmetric positive definite matrix')
    if not is_mat_symm(b_mat):
        throw_error('wrongInput:b_mat', 'second argument must be a symmetric matrix')
    if a_mat.shape[0] != b_mat.shape[0]:
        throw_error('wrongInput', 'both matrices must be of the same dimension')

    u1_mat, s_vec, _ = np.linalg.svd(a_mat, full_matrices=True)
    u_mat = np.linalg.lstsq(sqrtm_pos(np.diag(s_vec), abs_tol).T, u1_mat.T, -1)[0].T
    u2_mat, _, _ = np.linalg.svd(u_mat.T @ b_mat @ u_mat)
    return u2_mat.T @ u_mat.T
