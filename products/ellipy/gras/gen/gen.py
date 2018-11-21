from typing import Tuple, Callable, Union
import numpy as np
from ellipy.gen.common.common import throw_error


def mat_dot(inp_arr1: np.ndarray, inp_arr2: np.ndarray) -> np.ndarray:
    pass


def sort_rows_tol(inp_mat: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass


def sqrt_pos(inp_arr: Union[int, float, np.ndarray], abs_tol: float = 0.) -> Union[float, np.ndarray]:
    if abs_tol < 0.:
        throw_error('wrongInput:abs_tolNegative', 'abs_tol is expected to be nonnegative')
    if np.isscalar(inp_arr):
        inp_arr_new = np.float64(inp_arr)
        if inp_arr_new < -abs_tol:
            throw_error('wrongInput:negativeInput', 'input value is under -abs_tol')
        elif inp_arr_new < 0.:
            inp_arr_new = 0.
        return np.sqrt(inp_arr_new)
    else:
        inp_arr_new = np.array(inp_arr, copy=True, dtype=np.float64)
        if np.any(inp_arr_new < -abs_tol):
            throw_error('wrongInput:negativeInput', 'input array contains values under -abs_tol')
        inp_arr_new[inp_arr_new < 0.] = 0.
        return np.sqrt(inp_arr_new)


class MatVector:
    @staticmethod
    def triu(data_arr: np.ndarray) -> np.ndarray:
        ret_arr = np.copy(data_arr)
        if len(data_arr.shape) == 2:
            ret_arr = np.expand_dims(ret_arr, 2)
        arr_size = ret_arr.shape[2]
        for i_elem in range(arr_size):
            ret_arr[:, :, i_elem] = np.triu(ret_arr[:, :, i_elem])
        return ret_arr

    @staticmethod
    def make_symmetric(data_arr: np.ndarray) -> np.ndarray:
        ret_arr = 0.5 * (data_arr + MatVector.transpose(data_arr));
        return ret_arr

    @staticmethod
    def pinv(data_arr: np.ndarray) -> np.ndarray:
        arr_size = data_arr.shape
        if len(arr_size) == 2:
            arr_size = (arr_size[0], arr_size[1], 1)
        inv_data_array = np.zeros((arr_size[1], arr_size[0], arr_size[2]))
        for t in range(arr_size[2]):
            inv_data_array[:, :, t] = np.linalg.pinv(data_arr[:, :, t])
        return inv_data_array

    @staticmethod
    def transpose(inp_arr: np.ndarray) -> np.ndarray:
        inp_size = inp_arr.shape
        data_arr = np.copy(inp_arr)
        if len(inp_size) == 2:
            data_arr = np.expand_dims(data_arr, 2)
        trans_arr = np.transpose(data_arr, (1, 0, 2))
        return trans_arr

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
    def from_expression(exp_str: str, t_vec: Union[int, float, np.ndarray]) -> np.ndarray:
        exec("from numpy import *")
        t = np.asarray(t_vec, dtype=np.float64).flatten()
        if len(t) == 1:
            ret_val = np.array(eval(exp_str))
            if len(ret_val.shape) < 3:
                ret_val = np.expand_dims(ret_val, 2)
            return ret_val
        else:
            time_len = len(t)
            ret_val = eval(exp_str)
            ret_val = [[np.repeat(np.asarray(j), time_len) if type(j) != np.ndarray else j
                        for j in i]
                       for i in ret_val]
            ret_val = np.array(ret_val)
            if len(ret_val.shape) < 3:
                ret_val = np.expand_dims(ret_val, 2)
                ret_val = np.tile(ret_val, (1, 1, len(t)))
            return ret_val

    @staticmethod
    def r_multiply_by_vec(a_arr: np.ndarray, b_mat: np.ndarray, use_sparse_matrix: bool = True) -> np.ndarray:
        from ellipy.gen.common.common import throw_error
        if len(b_mat.shape) != 2:
            throw_error('wrongInput', 'bMat is expected to be 2-dimensional array')
        n_rows = a_arr.shape[0]
        n_cols = a_arr.shape[1]
        n_time_points = a_arr.shape[2]
        if use_sparse_matrix:
            pass
        else:
            c_mat = np.zeros(n_rows, n_time_points)
            for i_time_point in range(0, n_time_points):
                c_mat[:, i_time_point] = a_arr[:, :, i_time_point] @ b_mat[:, i_time_point]
        return c_mat

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
    def make_pos_definite_by_eig(data_arr: np.ndarray, value: float = 1e-12) -> np.ndarray:
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
