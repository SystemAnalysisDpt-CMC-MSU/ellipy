from typing import Tuple, Callable, Union, List
import numpy as np
import scipy.sparse as sp
from ellipy.gen.common.common import throw_error, is_numeric
from numpy import linalg as la
import os
import scipy.io


def mat_dot(inp_arr1: np.ndarray, inp_arr2: np.ndarray) -> np.ndarray:
    pass


def sort_rows_tol(inp_mat: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tol < 0.:
        throw_error('wrongInput:tol', 'tol is expected to be a positive number')
    if not (inp_mat.ndim == 2 and is_numeric(inp_mat)):
        throw_error('wrongInput:inp_mat', 'input is expected to be a numeric matrix')
    copy_inp_mat = np.copy(inp_mat)
    n_cols = np.size(copy_inp_mat, 1)
    n_rows = np.size(copy_inp_mat, 0)
    if n_rows > 0:
        res_mat = np.copy(copy_inp_mat)

        for i_col in range(n_cols):
            ind_col_sort_vec = np.argsort(copy_inp_mat[:, i_col])
            col_vec = copy_inp_mat[ind_col_sort_vec, i_col]
            col_diff_vec = np.diff(col_vec)
            is_less_vec = np.abs(col_diff_vec) <= tol
            col_diff_vec[is_less_vec] = 0.
            col_vec = np.cumsum(np.hstack((col_vec[0], col_diff_vec)))
            ind_col_rev_sort_vec = np.argsort(ind_col_sort_vec)
            copy_inp_mat[:, i_col] = col_vec[ind_col_rev_sort_vec]

        ind_sort_vec = np.lexsort(np.fliplr(copy_inp_mat).T)
        res_mat = res_mat[ind_sort_vec]
    else:
        res_mat = copy_inp_mat
        ind_sort_vec = np.empty((0, ), dtype=np.float64)

    ind_rev_sort_vec = np.argsort(ind_sort_vec)
    return res_mat, ind_sort_vec, ind_rev_sort_vec


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
    def __to_array(inp_vec: Union[int, float, np.ndarray]) -> np.ndarray:
        ret = np.asarray(inp_vec, dtype=np.float64).flatten()
        return ret

    @staticmethod
    def triu(data_arr: np.ndarray) -> np.ndarray:
        ret_arr = np.copy(data_arr)
        if data_arr.ndim == 2:
            ret_arr = np.expand_dims(ret_arr, 2)
        arr_size = ret_arr.shape[2]
        for i_elem in range(arr_size):
            ret_arr[:, :, i_elem] = np.triu(ret_arr[:, :, i_elem])
        return ret_arr

    @staticmethod
    def make_symmetric(data_arr: np.ndarray) -> np.ndarray:
        ret_arr = 0.5 * (data_arr + MatVector.transpose(data_arr))
        return ret_arr

    @staticmethod
    def pinv(data_arr: np.ndarray) -> np.ndarray:
        arr_size = data_arr.shape
        if data_arr.ndim == 2:
            arr_size = (arr_size[0], arr_size[1], 1)
        inv_data_array = np.zeros((arr_size[1], arr_size[0], arr_size[2]), dtype=np.float64)
        for t in range(arr_size[2]):
            inv_data_array[:, :, t] = np.linalg.pinv(data_arr[:, :, t])
        return inv_data_array

    @staticmethod
    def transpose(inp_arr: np.ndarray) -> np.ndarray:
        data_arr = np.copy(inp_arr)
        if inp_arr.ndim == 2:
            data_arr = np.expand_dims(data_arr, 2)
        trans_arr = np.transpose(data_arr, (1, 0, 2))
        return trans_arr

    @staticmethod
    def from_formula_mat(x: np.ndarray, t_vec: Union[int, float, np.ndarray]) -> np.ndarray:
        data = np.copy(x)
        data_shape = data.shape
        if len(data_shape) < 2:
            data = np.expand_dims(data, 1)
            data_shape += (1,)
        t = MatVector.__to_array(t_vec)
        ret_arr = np.zeros(data.shape + t.shape)
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                ret_arr[i, j, :] = MatVector.from_expression(data[i, j], t_vec)[0, 0, :]
        return ret_arr

    @staticmethod
    def from_func(f: Callable[[float], np.ndarray], t_vec: Union[int, float, np.ndarray]) -> np.ndarray:
        t = MatVector.__to_array(t_vec)
        n_time_points = t.size
        first_val = f(t[0])
        if type(first_val) != np.ndarray:
            first_val = MatVector.__to_array(first_val)
        if first_val.ndim < 2:
            first_val = np.expand_dims(first_val, 1)
        size = first_val.shape
        ret_val = np.zeros((size[0], size[1], n_time_points), dtype=np.float64)
        ret_val[:, :, 0] = first_val
        for i in range(1, n_time_points):
            # we have to be sure that the result is in fact a matrix
            # in case is't a vector (that is, res.shape = (n,) we expand
            # it into a (n, 1) vector
            f_val = f(t[i])
            if type(f_val) != np.ndarray:
                f_val = MatVector.__to_array(f_val)
            if f_val.ndim < 2:
                f_val = np.expand_dims(f_val, 1)
            ret_val[:, :, i] = f_val
        return ret_val

    @staticmethod
    def eval_func(f: Callable[[np.ndarray], np.ndarray], inp_arr: np.ndarray,
                  uniform_output: bool = True, keep_size: bool = False) \
            -> Union[np.ndarray, List[np.ndarray]]:
        data_arr = np.copy(inp_arr)
        data_shape = data_arr.shape
        if data_arr.ndim < 3:
            data_arr = np.expand_dims(data_arr, 2)
            data_size = 1
        else:
            data_size = data_shape[2]
        if not uniform_output:
            res_array = list()
            for i in range(data_size):
                res_array.append(f(data_arr[:, :, i]))
        else:
            if keep_size:
                res_array = np.copy(data_arr)
                for i in range(data_size):
                    res_array[:, :, i] = f(data_arr[:, :, i])
            else:
                res_array = np.zeros(data_size, dtype=np.float64)
                for i in range(data_size):
                    res_array[i] = f(data_arr[:, :, i])
        return res_array

    @staticmethod
    def from_expression(expr: str, t_vec: Union[int, float, np.ndarray]) -> np.ndarray:
        exec("from numpy import *")
        # string check
        exp_str = expr
        if len(exp_str) < 1:
            throw_error('wrongInput', 'expr must be a non-empty string!')
        if exp_str[0] != "[":
            exp_str = "[[" + exp_str
        elif exp_str[1] != "[":
            exp_str = "[" + exp_str
        if exp_str[-1] != "]":
            exp_str = exp_str + "]]"
        elif exp_str[-2] != "]":
            exp_str = exp_str + "]"
        t = MatVector.__to_array(t_vec)
        if len(t) == 1:
            ret_val = np.array(eval(exp_str))
            if ret_val.ndim < 2:
                ret_val = np.expand_dims(ret_val, 1)
            if ret_val.ndim < 3:
                ret_val = np.expand_dims(ret_val, 2)
            return ret_val
        else:
            time_len = len(t)
            ret_val = eval(exp_str)
            ret_val = [[np.repeat(np.asarray(j), time_len) if type(j) != np.ndarray else j
                        for j in i]
                       for i in ret_val]
            ret_val = np.array(ret_val)
            ret_shape = ret_val.shape
            if len(ret_shape) < 3:
                ret_val = np.expand_dims(ret_val, 0)
            return ret_val

    @staticmethod
    def r_multiply_by_vec(a_arr: np.ndarray, b_mat: np.ndarray, use_sparse_matrix: bool = True) -> np.ndarray:
        if len(b_mat.shape) != 2:
            throw_error('wrongInput', 'b_mat is expected to be 2-dimensional array')
        n_rows = a_arr.shape[0]
        n_cols = a_arr.shape[1]
        n_time_points = a_arr.shape[2]
        if use_sparse_matrix:
            i_ind = np.arange(n_cols * n_time_points)
            j_ind = np.repeat(np.arange(n_time_points), n_cols)
            b_sparse = sp.csc_matrix((b_mat.T.flatten(), (i_ind, j_ind)), shape=(n_cols * n_time_points, n_time_points))
            ret_mat = a_arr.reshape(n_rows, n_cols * n_time_points, order='F') @ b_sparse
        else:
            ret_mat = np.zeros((n_rows, n_time_points), dtype=np.float64)
            for i_time_point in range(n_time_points):
                ret_mat[:, i_time_point] = a_arr[:, :, i_time_point] @ b_mat[:, i_time_point]
        return ret_mat

    @staticmethod
    def r_multiply(a_arr: np.ndarray, b_arr: np.ndarray, c_arr: np.array = None,
                   use_sparse_matrix: bool = False) -> np.ndarray:
        def get_sparse_mat(inp_arr: np.ndarray) -> sp.csc_matrix:
            int_arr = inp_arr
            inp_shape = inp_arr.shape
            if inp_arr.ndim < 3:
                int_arr = np.tile(np.expand_dims(int_arr, 2), (1, 1, n_time_points))
            n_rows = inp_shape[0]
            n_cols = inp_shape[1]
            i_mat = np.tile(np.arange(n_rows * n_time_points), (n_cols, 1))
            i_mat = np.reshape(i_mat, (n_cols, n_rows, n_time_points), order='F')
            i_mat = np.transpose(i_mat, (1, 0, 2))
            i_ind = np.reshape(i_mat, (n_rows * n_cols * n_time_points), order='F')
            j_ind = np.tile(np.arange(n_cols * n_time_points), (n_rows, 1))
            j_ind = np.reshape(j_ind, (n_rows * n_cols * n_time_points), order='F')
            return sp.csc_matrix((int_arr.T.flatten(), (i_ind, j_ind)),
                                 shape=(n_rows * n_time_points, n_cols * n_time_points))

        use_sparse = use_sparse_matrix
        a_shape = a_arr.shape
        n_a_rows = a_shape[0]
        n_a_cols = a_shape[1]
        n_time_points = a_shape[2]
        b_shape = b_arr.shape
        n_b_rows = b_shape[0]
        n_b_cols = b_shape[1]
        if b_arr.ndim < 3:
            b_shape += (1,)
        if c_arr is None:
            is_binary = True
            n_c_rows = None
            n_c_cols = None
        else:
            c_shape = c_arr.shape
            n_c_rows = c_shape[0]
            n_c_cols = c_shape[1]
            is_binary = False
        is_a_scalar = (n_a_cols == 1) and (n_a_rows == 1)
        is_b_scalar = (n_b_cols == 1) and (n_b_rows == 1)
        if is_a_scalar or is_b_scalar:
            use_sparse = False
        if use_sparse:
            a_mat = np.reshape(a_arr, (n_a_rows, n_a_cols * n_time_points), order='F')
            if is_binary:
                res_mat = a_mat @ get_sparse_mat(b_arr)
                res_mat = np.reshape(res_mat, (n_a_rows, n_b_cols, n_time_points), order='F')
            else:
                res_mat = a_mat @ get_sparse_mat(b_arr) @ get_sparse_mat(c_arr)
                res_mat = np.reshape(res_mat, (n_a_rows, n_c_cols, n_time_points), order='F')
        else:
            if is_binary:
                if is_a_scalar:
                    res_mat = np.zeros((n_b_rows, n_b_cols, n_time_points), dtype=np.float64)
                elif is_b_scalar:
                    res_mat = np.zeros((n_a_rows, n_a_cols, n_time_points), dtype=np.float64)
                else:
                    res_mat = np.zeros((n_a_rows, n_b_cols, n_time_points), dtype=np.float64)
                if b_shape[2] == n_time_points:
                    for i_time_point in range(n_time_points):
                        res_mat[:, :, i_time_point] = a_arr[:, :, i_time_point] @ b_arr[:, :, i_time_point]
                elif b_shape[2] == 1:
                    for i_time_point in range(n_time_points):
                        res_mat[:, :, i_time_point] = a_arr[:, :, i_time_point] @ b_arr
                else:
                    throw_error('wrongInput', 'Incorrect size of b_arr')
            else:
                if is_a_scalar and is_b_scalar:
                    res_mat = np.zeros((n_c_rows, n_c_cols, n_time_points), dtype=np.float64)
                elif is_a_scalar:
                    res_mat = np.zeros((n_b_rows, n_c_cols, n_time_points), dtype=np.float64)
                else:
                    res_mat = np.zeros((n_a_rows, n_c_cols, n_time_points), dtype=np.float64)
                for i_time_point in range(n_time_points):
                    res_mat[:, :, i_time_point] = a_arr[:, :, i_time_point] @ \
                                                  b_arr[:, :, i_time_point] @ c_arr[:, :, i_time_point]
        return res_mat


class SquareMatVector(MatVector):
    @staticmethod
    def inv(data_arr: np.ndarray) -> np.ndarray:
        dim_num = data_arr.ndim
        if dim_num == 2:
            inv_data_array = np.linalg.inv(data_arr)
        else:
            size_vec = data_arr.shape
            inv_data_array = np.zeros(size_vec)
            for t in range(size_vec[2]):
                inv_data_array[:, :, t] = np.linalg.inv(data_arr[:, :, t])
        return inv_data_array

    @staticmethod
    def sqrtm_pos(data_arr: np.ndarray) -> np.ndarray:
        from ellipy.gras.la.la import sqrtm_pos
        dim_num = data_arr.ndim
        if dim_num == 2:
            sqrt_data_array = sqrtm_pos(data_arr)
        else:
            size_vec = data_arr.shape
            sqrt_data_array = np.zeros(size_vec)
            for t in range(size_vec[2]):
                if np.isnan(data_arr[:, :, t]).any():
                    sqrt_data_array[:, :, t] = np.NaN
                sqrt_data_array[:, :, t] = sqrtm_pos(np.squeeze(data_arr[:, :, t]))
        return sqrt_data_array

    @staticmethod
    def make_pos_definite_or_nan(data_arr: np.ndarray) -> np.ndarray:
        dim_num = data_arr.ndim
        if dim_num == 2:
            s_min = min(np.linalg.eigvals(data_arr))
            if s_min < 0:
                res_data_arr = np.nan
            else:
                res_data_arr = data_arr
        else:
            size_vec = data_arr.shape
            res_data_arr = np.zeros(size_vec)
            for t in range(size_vec[2]):
                if min(np.linalg.eigvals(data_arr[:, :, t])) < 0:
                    res_data_arr[:, :, t] = np.nan
                else:
                    res_data_arr[:, :, t] = data_arr[:, :, t]
        return res_data_arr

    @staticmethod
    def make_pos_definite_by_eig(data_arr: np.ndarray, value: float = 1e-12) -> np.ndarray:
        from ellipy.gras.la.la import is_mat_symm
        dim_num = data_arr.ndim
        size_vec = data_arr.shape
        res_data_arr = np.zeros(size_vec)
        if dim_num == 2:
            if not is_mat_symm(data_arr):
                throw_error('wrongInput:non SymmMat', 'input matrix must be symetric')
            d, v = np.linalg.eigh(data_arr)
            d[d < 0] = value
            res_data_arr = (v @ np.diag(d) @ v.T).real
        else:
            for t in range(size_vec[2]):
                if not is_mat_symm(data_arr[:, :, t]):
                    throw_error('wrongInput:non SymmMat', 'input matrix must be symetric')
                d, v = np.linalg.eigh(data_arr[:, :, t])
                d[d < 0] = value
                res_data_arr[:, :, t] = (v @ np.diag(d) @ v.T).real
        return res_data_arr

    @staticmethod
    def lr_multiply(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray, flag: str = 'R') -> np.ndarray:
        a_size_vec = inp_a_arr.shape
        b_size_vec = inp_b_arr.shape
        if inp_b_arr.ndim == 2:
            if len(inp_a_arr.shape) <= 2:
                if flag == 'R':
                    out_array = inp_a_arr.T@inp_b_arr@inp_a_arr
                elif flag == 'L':
                    out_array = inp_a_arr@inp_b_arr@inp_a_arr.T
                else:
                    throw_error('wrong_input', 'flag ' + flag + ' is not supported')
            else:
                if flag == 'R':
                    out_array = inp_a_arr[:, :, 0].T@inp_b_arr@inp_a_arr[:, :, 0]
                elif flag == 'L':
                    out_array = inp_a_arr[:, :, 0]@inp_b_arr@inp_a_arr[:, :, 0].T
                else:
                    throw_error('wrong_input', 'flag ' + flag + ' is not supported')
        else:
            if len(inp_a_arr.shape) <= 2:
                if flag == 'R':
                    out_array = np.zeros((a_size_vec[1], a_size_vec[1], b_size_vec[2]))
                    for t in range(b_size_vec[2]):
                        out_array[:, :, t] = inp_a_arr.T @ inp_b_arr[:, :, t] @ inp_a_arr
                elif flag == 'L':
                    out_array = np.zeros((a_size_vec[0], a_size_vec[0], b_size_vec[2]))
                    for t in range(b_size_vec[2]):
                        out_array[:, :, t] = inp_a_arr @ inp_b_arr[:, :, t] @ inp_a_arr.T
                else:
                    throw_error('wrong_input', 'flag ' + flag + ' is not supported')
            else:
                if flag == 'R':
                    out_array = np.zeros((a_size_vec[1], a_size_vec[1], b_size_vec[2]))
                    for t in range(b_size_vec[2]):
                        out_array[:, :, t] = inp_a_arr[:, :, t].T @ inp_b_arr[:, :, t] @ inp_a_arr[:, :, t]
                elif flag == 'L':
                    out_array = np.zeros((a_size_vec[0], a_size_vec[0], b_size_vec[2]))
                    for t in range(b_size_vec[2]):
                        out_array[:, :, t] = inp_a_arr[:, :, t] @ inp_b_arr[:, :, t] @ inp_a_arr[:, :, t].T
                else:
                    throw_error('wrong_input', 'flag ' + flag + ' is not supported')
        return out_array

    @staticmethod
    def lr_multiply_by_vec(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray) -> np.ndarray:
        if len(inp_a_arr.shape) == 1:
            inp_a_arr.shape = (inp_a_arr.size, 1)
        a_size_vec = inp_a_arr.shape
        if inp_b_arr.ndim == 2:
            out_vec = np.zeros((1, inp_a_arr.shape[1]))
            for t in range(a_size_vec[1]):
                out_vec[:, t] = inp_a_arr[:, t].T.dot(inp_b_arr.dot(inp_a_arr[:, t]))
        else:
            out_vec = np.zeros((1, inp_a_arr.shape[1]))
            for t in range(a_size_vec[1]):
                out_vec[:, t] = inp_a_arr[:, t].T.dot(inp_b_arr[:, :, t].dot(inp_a_arr[:, t]))
        return out_vec

    @staticmethod
    def lr_divide_vec(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray) -> np.ndarray:
        if len(inp_a_arr.shape) == 1:
            inp_a_arr.shape = (inp_a_arr.size, 1)
        if inp_b_arr.ndim == 2:
            out_vec = inp_a_arr.T@np.linalg.lstsq(inp_b_arr, inp_a_arr)[0]
        else:
            if inp_a_arr.shape[1] == 1:
                out_vec = np.zeros((inp_b_arr.shape[2],))
                for t in range(inp_b_arr.shape[2]):
                    out_vec[t] = inp_a_arr.T @ np.linalg.lstsq(inp_b_arr[:, :, t], inp_a_arr)[0]
            else:
                out_vec = np.zeros((inp_a_arr.shape[1],))
                for t in range(inp_a_arr.shape[1]):
                    out_vec[t] = inp_a_arr[:, t].T @ np.linalg.lstsq(inp_b_arr[:, :, t], inp_a_arr[:, t])[0]
        return out_vec


class SymmetricMatVector(SquareMatVector):

    @staticmethod
    def lr_svd_multiply(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray, flag: str = 'R') -> np.ndarray:
        u_array, s_array = SymmetricMatVector.__array_svd(inp_b_arr)
        ua_array = None
        if flag == 'L':
            ua_array = SquareMatVector.r_multiply(u_array, MatVector.transpose(inp_a_arr))
        elif flag == 'R':
            ua_array = SquareMatVector.r_multiply(u_array, inp_a_arr)
        else:
            throw_error('wrongInput:flag', 'flag %s is not supported' % flag)
        out_array = SquareMatVector.lr_multiply(s_array, ua_array, flag)
        return out_array

    @staticmethod
    def r_svd_multiply_by_vec(inp_mat_arr: np.ndarray, inp_vec_arr: np.ndarray) -> np.ndarray:
        u_array, s_array = SymmetricMatVector.__array_svd(inp_mat_arr)
        uv_array = MatVector.r_multiply_by_vec(u_array, inp_vec_arr)
        if inp_vec_arr.ndim != 2:
            throw_error('wrongInput:inp_vec_arr', 'inpVecArray is expected to be 2-dimensional array')
        m_size_vec = np.shape(s_array)
        v_size_vec = np.shape(uv_array)
        out_vec_array = np.zeros((m_size_vec[0], v_size_vec[1]), dtype=np.float64)
        for t in range(v_size_vec[1]):
            out_vec_array[:, t] = u_array[:, :, t].T @ s_array[:, :, t] @ uv_array[:, t]
        return out_vec_array

    @staticmethod
    def lr_svd_multiply_by_vec(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray) -> np.ndarray:
        u_array, s_array = SymmetricMatVector.__array_svd(inp_b_arr)
        ua_array = MatVector.r_multiply_by_vec(u_array, inp_a_arr)
        out_vec = SquareMatVector.lr_multiply_by_vec(s_array, ua_array)
        return out_vec

    @staticmethod
    def lr_svd_divide_vec(inp_b_arr: np.ndarray, inp_a_arr: np.ndarray) -> np.ndarray:
        u_array, s_array = SymmetricMatVector.__array_svd(inp_b_arr)
        ua_array = MatVector.r_multiply_by_vec(u_array, inp_a_arr)
        #
        a_size_vec = np.shape(inp_a_arr)
        n_elems = a_size_vec[1]
        #
        n_mat_elems = np.shape(inp_b_arr)
        n_mat_elems = n_mat_elems[2]
        if n_mat_elems == 1:
            b_inv_mat = np.diag(1 / np.diag(s_array))
            out_vec = np.sum(((b_inv_mat @ ua_array) * ua_array), axis=0)
        else:
            out_vec = np.zeros(n_elems, dtype=np.float64)
            for i_elem in range(n_elems):
                b_inv_mat = np.diag(1 / np.diag(s_array[:, :, i_elem]))
                out_vec[i_elem] = (ua_array[:, i_elem].T @ (b_inv_mat @ ua_array[:, i_elem]))
        return out_vec

    @staticmethod
    def __array_svd(sym_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        size_vec = np.shape(sym_arr)
        if len(size_vec) == 2:
            size_vec = np.append(size_vec, 1)
        u_array = np.zeros(size_vec, dtype=np.float64)
        s_array = np.zeros(size_vec, dtype=np.float64)
        for t in range(size_vec[2]):
            eig_vec, s_array[:, :, t] = la.eigh(sym_arr[:, :, t])
            u_array[:, :, t] = np.diag(eig_vec)
        return s_array, u_array
