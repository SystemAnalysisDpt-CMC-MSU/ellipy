from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.gras.la.la import *
from ellipy.gen.common.common import throw_error, is_numeric
from numpy.linalg import inv, det, svd
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


def ell_fusion_lambda(a, q1: np.ndarray, q1_mat: np.ndarray, q2: np.ndarray, q2_mat: np.ndarray, n) -> float:
    x_mat = a * q1_mat + (1 - a) * q2_mat
    y_mat = inv(x_mat)
    y_mat = 0.5 * (y_mat + y_mat.T)
    k = 1 - a * (1 - a) * (q2 - q1).T @ q2_mat @ y_mat @ q1_mat @ (q2 - q1)
    q = y_mat @ (a * q1_mat @ q1 + (1 - a) * q2_mat @ q2)

    f = k * det(x_mat) * np.trace(det(x_mat) * y_mat @ (q1_mat - q2_mat)) - \
        n * ((det(x_mat)) ** 2) * (2 * q.T @ q1_mat @ q1 - 2 * q.T @ q2_mat @ q2 +
                                   q.T @ (q2_mat - q1_mat) @ q - q1.T @ q1_mat @ q1 + q2.T @ q2_mat @ q2)

    return f


def ell_valign(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    if (not is_numeric(v)) or (not is_numeric(x)):
        throw_error('wrongInput:v,x', 'ELL_VALIGN: both arguments must be vectors in R^n.')

    if v.ndim != 2:
        throw_error('wrongInput:v', 'ELL_VALIGN: first argument must be 2-dimension vector.')
    if x.ndim != 2:
        throw_error('wrongInput:x', 'ELL_VALIGN: second argument must be 2-dimension vector.')

    v_dim1, v_dim2 = v.shape
    x_dim1, x_dim2 = x.shape

    if (v_dim2 != 1) or (x_dim2 != 1):
        throw_error('wrongInput:v,x', 'ELL_VALIGN: both arguments must be vectors in R^n.')

    if v_dim1 != x_dim1:
        throw_error('wrongInput:v,x', 'ELL_VALIGN: both vectors must be of the same dimension.')

    u1_mat, _, v1_mat = svd(v)
    u2_mat, _, v2_mat = svd(x)

    v2_mat = v2_mat[0, 0]
    v1_mat = v1_mat[0, 0]

    t_mat = v1_mat * v2_mat * u1_mat @ u2_mat.T

    return t_mat
