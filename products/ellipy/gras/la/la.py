from ellipy.gen.common.common import throw_error, abs_rel_compare
from typing import Callable
import numpy as np
from numpy import linalg


def is_mat_not_deg(q_mat: np.ndarray, abs_tol: float) -> bool:
    pass


def is_mat_pos_def(q_mat: np.ndarray, abs_tol: float = 0., is_flag_sem_def_on: bool = False) -> bool:
    pass


def is_mat_symm(q_mat: np.ndarray, abs_tol: float = 0.) -> bool:
    if q_mat.ndim != 2:
        throw_error('wrongInput:nonSquareMat', 'q_mat should be a matrix')
    n_rows = q_mat.shape[0]
    n_cols = q_mat.shape[1]
    if n_rows != n_cols:
        throw_error('wrongInput:nonSquareMat', 'q_mat should be a square matrix')

    abs_func: Callable[[np.ndarray], np.ndarray] = lambda x: np.abs(x)
    is_symm, *_ = abs_rel_compare(q_mat, q_mat.T, abs_tol, None, abs_func)
    return is_symm


def mat_orth(src_mat: np.ndarray) -> np.ndarray:
    pass


def math_orth_col(src_mat: np.ndarray) -> np.ndarray:
    pass


def ml_orth_transl(src_mat: np.ndarray, dst_arr: np.ndarray) -> np.ndarray:
    pass


def orth_transl(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
    abs_tol = 1e-7
    n_dims = dst_vec.shape[0]
    src_vec = try_treat_as_real(src_vec)
    dst_vec = try_treat_as_real(dst_vec)
    dst_squared_norm = np.sum(dst_vec * dst_vec)
    src_squared_norm = np.sum(src_vec * src_vec)

    if dst_squared_norm == 0:
        throw_error('wrongInput:dstZero', 'destination vectors are expected to be non-zero')
    if src_squared_norm == 0:
        throw_error('wrongInput:srcZero', 'source vectors are expected to be non-zero')

    dst_vec = dst_vec / np.sqrt(dst_squared_norm)
    src_vec = src_vec / np.sqrt(src_squared_norm)

    scal_prod = np.sum(src_vec * dst_vec)
    s_val = np.sqrt(max(1 - scal_prod * scal_prod, 0))
    q_mat = np.zeros((n_dims, 2), dtype=float)
    q_mat[:, 0] = dst_vec[:, 0]
    if np.abs(s_val) > abs_tol:
        q_mat[:, 1] = ((src_vec - scal_prod * dst_vec) / s_val)[:, 0]
    else:
        q_mat[:, 1] = 0

    s_mat = [[scal_prod - 1, s_val], [-s_val, scal_prod - 1]]
    o_mat = np.identity(n_dims, dtype=float) + np.dot(np.dot(q_mat, s_mat), q_mat.transpose())
    return o_mat


def orth_transl_haus(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
    pass


def orth_transl_max_dir(src_vec: np.ndarray, dst_vec: np.ndarray,
                        src_max_vec: np.ndarray, dst_max_vec: np.ndarray) -> np.ndarray:
    pass


def orth_transl_max_tr(src_vec: np.ndarray, dst_vec: np.ndarray, max_mat: np.ndarray) -> np.ndarray:
    pass


def orth_transl_qr(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
    pass


def reg_mat(inp_mat: np.ndarray, reg_tol: float) -> np.ndarray:
    pass


def reg_pos_def_mat(inp_mat: np.ndarray, reg_tol: float) -> np.ndarray:
    if not(np.isscalar(reg_tol) and is_numeric(reg_tol) and reg_tol > 0):
        throw_error('wrongInput:reg_tol', 'reg_tol must be a positive numeric scalar')
    reg_tol = try_treat_as_real(reg_tol)
    if not(is_mat_symm(inp_mat)):
        throw_error('wrongInput:inp_mat', 'matrix must be symmetric')
    d_mat, v_mat = np.linalg.eig(inp_mat)
    m_mat = np.diag(np.maximum(0, reg_tol-d_mat))
    m_mat = np.dot(np.dot(v_mat, m_mat), v_mat.transpose())
    regular_mat = inp_mat + m_mat
    regular_mat = 0.5 * (regular_mat + regular_mat.transpose())
    return regular_mat


def sqrtm_pos(q_mat: np.ndarray, abs_tol: float) -> np.ndarray:
    pass


def try_treat_as_real(inp_mat:  np.ndarray, tol_val: float = np.finfo(float).eps) -> np.ndarray:
    if not(np.isscalar(tol_val) and is_numeric(tol_val) and tol_val > 0):
        throw_error('wrongInput:tol_val', 'tol_val must be a positive numeric scalar')
    if np.all(np.isreal(inp_mat)):
        return inp_mat
    else:
        img_inp_mat = inp_mat.imag
        norm_value = linalg.norm(img_inp_mat, np.inf)
        if norm_value < tol_val:
            return inp_mat.real
        else:
            out_vec = str('Norm of imaginary part of source object = ' + str(norm_value) +
                          '. It can not be more than tol_val = ' + str(tol_val))
            throw_error('wrongInput:inp_mat', out_vec)

