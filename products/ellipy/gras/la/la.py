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
    pass


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
    pass


def sqrtm_pos(q_mat: np.ndarray, abs_tol: float) -> np.ndarray:
    pass


def try_treat_as_real(inp_mat:  np.ndarray, tol_val: float = np.finfo(float).eps) -> np.ndarray:
    if not(np.isscalar(tol_val) and is_numeric(tol_val) and tol_val > 0):
        throw_error('wrongInput:tolVal', 'tolVal must be a positive numeric scalar')
    if np.all(np.isreal(inp_mat)):
        return inp_mat
    else:
        img_inp_mat = inp_mat.imag
        norm_value = linalg.norm(img_inp_mat, np.inf)
        if norm_value < tol_val:
            return inp_mat.real
        else:
            out_vec = str('Norm of imaginary part of source object = ' + str(norm_value) +
                          '. It can not be more than tolVal = ' + str(tol_val))
            throw_error('wrongInput:inpMat', out_vec)
