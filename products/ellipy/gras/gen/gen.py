from typing import Tuple, Callable
import numpy as np
from ellipy.gen.common.common import throw_error

def mat_dot(inp_arr1: np.ndarray, inp_arr2: np.ndarray) -> np.ndarray:
    pass


def sort_rows_tol(inp_mat: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass


def sqrt_pos(inp_arr: np.ndarray, abs_tol: float = 0.) -> np.ndarray:
    if abs_tol < 0.:
        throw_error('wrongInput:abs_tolNegative', 'abs_tol is expected to be not-negative')
    if np.isscalar(inp_arr):
        if inp_arr < -abs_tol:
            throw_error('wrongInput:negativeInput', 'input value is under -abs_tol')
        elif inp_arr < 0.:
            inp_arr = 0.
        return np.sqrt(inp_arr)
    else:
        if inp_arr.any() < -abs_tol:
            throw_error('wrongInput:negativeInput', 'input array contains values under -abs_tol')
        inp_arr[inp_arr < 0.] = 0.
        return np.sqrt(inp_arr)


class MatVector:
    @staticmethod
    def triu(data_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def make_symmetric(data_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def pinv(data_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def transpose(inp_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def from_formula_mat(x: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def from_func(f: Callable[[float], np.ndarray], t_vec: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def eval_func(f: Callable[[np.ndarray], np.ndarray], data_arr: np.ndarray,
                  uniform_output: bool = True, keep_size: bool = False) -> np.ndarray:
        pass

    @staticmethod
    def from_expression(exp_str: str, t_vec: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def r_multiply_by_vec(a_arr: np.ndarray, b_mat: np.ndarray, use_sparse_matrix: bool = True) -> np.ndarray:
        pass

    @staticmethod
    def r_multiply(a_arr: np.ndarray, b_arr: np.ndarray, c_arr: np.array,
                   use_sparse_matrix: bool = False) -> np.ndarray:
        pass


class SquareMatVector(MatVector):
    @staticmethod
    def inv(data_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def sqrtm_pos(data_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def make_pos_definite_or_nan(data_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def make_pos_definite_by_eig(data_arr: np.ndarray, value: float=1e-12) -> np.ndarray:
        pass

    @staticmethod
    def lr_multiply(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray, flag: str = 'R') -> np.ndarray:
        pass

    @staticmethod
    def lr_multiply_by_vec(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def lr_divide_vec(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray) -> np.ndarray:
        pass


class SymmetricMatVector(SquareMatVector):
    @staticmethod
    def __array_svd(sym_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def lr_svd_multiply(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray, flag: str = 'R') -> np.ndarray:
        pass

    @staticmethod
    def r_svd_multiply_by_vec(inp_mat_arr: np.ndarray, inp_vec_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def lr_svd_multiply_by_vec(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def lr_svd_divide_vec(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray) -> np.ndarray:
        pass
