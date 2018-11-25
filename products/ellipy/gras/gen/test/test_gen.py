from ellipy.gras.gen.gen import *
from typing import List
import numpy as np
import scipy.io
import os
import pytest
from numpy import linalg as la


class TestGen:
    __SQUARE_MAT_VEC_DATA = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'square_mat_vec_data.mat'))['res_struct']
    def test_sqrt_pos(self):
        def is_not_neg(*args):
            try:
                is_not_neg_arr = sqrt_pos(*args)
                is_not_neg_ans = np.all(is_not_neg_arr.reshape((1, -1)))
            except Exception as e:
                is_neg = str(e) + 'wrongInput:negativeInput'
                if not is_neg:
                    raise is_neg
                is_not_neg_ans = not is_neg
            return is_not_neg_ans

        def check(*args):
            inp_arr_f, *_ = args
            out_arr = sqrt_pos(*args)
            assert np.all(out_arr.shape == inp_arr_f.shape)
            res_arr = sqrt_pos(*args)
            exp_res_arr = np.array([sqrt_pos(x) for x in inp_arr_f.reshape((1, -1))])
            assert np.array_equal(res_arr.reshape(1, -1), exp_res_arr)
            is_not_neg_arr = np.array([sqrt_pos(x) for x in inp_arr_f.reshape((1, -1))])
            is_exp_not_neg = np.all(is_not_neg_arr.reshape((1, -1)))
            is_not_neg_res = is_not_neg(*args)
            assert np.array_equal(is_not_neg_res, is_exp_not_neg)

        inp_arr = np.random.rand(2, 3, 4, 5, 6)
        check(inp_arr)
        check(inp_arr.reshape((1, -1))[:, 100])
        inp_arr = np.eye(300)
        check(inp_arr)
        inp_arr = np.array([[1]])
        check(inp_arr)
        check(np.array([[2]]))

    def test_triu_single(self):
        data_arr = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
        exp_arr = np.array([[[1], [2], [3]], [[0], [5], [6]], [[0], [0], [9]]])
        res_arr = MatVector.triu(data_arr)
        assert np.array_equal(res_arr, exp_arr)

    def test_triu_single_2d(self):
        data_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        exp_arr = np.array([[[1], [2], [3]], [[0], [5], [6]], [[0], [0], [9]]])
        res_arr = MatVector.triu(data_arr)
        assert np.array_equal(res_arr, exp_arr)

    def test_make_symmetric_short(self):
        data_arr = np.array([[[1], [2]], [[0], [1]]])
        exp_arr = np.array([[[1], [1]], [[1], [1]]])
        res_arr = MatVector.make_symmetric(data_arr)
        assert np.array_equal(res_arr, exp_arr)

    def test_make_symmetric_long(self):
        data_arr = np.array([[[1, 4], [2, 3]], [[0, 1], [1, 5]]])
        exp_arr = np.array([[[1, 4], [1, 2]], [[1, 2], [1, 5]]])
        res_arr = MatVector.make_symmetric(data_arr)
        assert np.array_equal(res_arr, exp_arr)

    def test_transpose_single(self):
        data_arr = np.array([[[1], [2]], [[0], [1]]])
        exp_arr = np.array([[[1], [0]], [[2], [1]]])
        res_arr = MatVector.transpose(data_arr)
        assert np.array_equal(res_arr, exp_arr)

    def test_transpose_single_2d(self):
        data_arr = np.array([[1, 2], [0, 1]])
        exp_arr = np.array([[[1], [0]], [[2], [1]]])
        res_arr = MatVector.transpose(data_arr)
        assert np.array_equal(res_arr, exp_arr)

    def test_transpose_long(self):
        data_arr = np.array([[[1, 5], [2, 6]], [[0, 7], [1, 8]]])
        exp_arr = np.array([[[1, 5], [0, 7]], [[2, 6], [1, 8]]])
        res_arr = MatVector.transpose(data_arr)
        assert np.array_equal(res_arr, exp_arr)

    def test_from_formula_mat(self):
        exp = np.array(["2 + t"])
        t = 1.
        res_arr = MatVector.from_formula_mat(exp, t)
        exp_arr = np.array([[[3.]]])
        assert np.array_equal(res_arr, exp_arr)
        t = np.array([1, 2, 3])
        res_arr = MatVector.from_formula_mat(exp, t)
        exp_arr = np.array([[[3., 4., 5.]]])
        assert np.array_equal(res_arr, exp_arr)
        exp = np.array([['cos(t)', '-sin(t)'], ['sin(t)', '-cos(t)']])
        t = np.array([0, np.pi])
        res_arr = MatVector.from_formula_mat(exp, t)
        exp_arr = np.array([[[1, -1], [0, 0]], [[0, 0], [-1, 1]]], dtype=np.float64)
        assert np.allclose(res_arr, exp_arr)

    def test_from_func_point(self):
        t = 0
        exp_arr = np.array([[[1.]]])
        res_arr = MatVector.from_func(lambda a: np.cos(a), t)
        assert np.allclose(res_arr, exp_arr)
        t = 1.
        exp_arr = np.array([[[3.5]]])
        res_arr = MatVector.from_func(lambda a: 2 * a + 1.5, t)
        assert np.allclose(res_arr, exp_arr)
        t = np.array([3, 4])
        exp_arr = np.array([[[6., 12.]]])
        res_arr = MatVector.from_func(lambda a: a ** 2 - a, t)
        assert np.allclose(res_arr, exp_arr)

    def test_from_func_matrix(self):
        t = np.array([[0, np.pi]])
        exp_arr = np.array([[[0., 0.], [1., -1.]]])
        res_arr = MatVector.from_func(lambda _t: np.array([[np.sin(_t), np.cos(_t)]]), t)
        assert np.allclose(res_arr, exp_arr)
        res_arr = MatVector.from_func(lambda _t: np.array([np.sin(_t), np.cos(_t)]), t)
        exp_arr = np.array([[[0., 0.]], [[1., -1.]]])
        assert np.allclose(res_arr, exp_arr)

    def test_eval_func(self):
        def mat_min(inp_arr: np.ndarray) -> np.ndarray:
            ret_val = np.copy(inp_arr)
            return -ret_val

        def list_equal(l: List[np.ndarray], r: List[np.ndarray]) -> bool:
            if len(l) != len(r):
                return False
            for i in range(len(l)):
                if not np.array_equal(l[i], r[i]):
                    return False
            return True

        # uniform keep size
        data_vec = np.array([[[1, -1], [2, -2]], [[3, -3.], [4., -4]]], dtype=np.float64)
        res_vec = MatVector.eval_func(mat_min, data_vec, True, True)
        exp_vec = np.array([[[-1., 1.], [-2., 2.]], [[-3., 3.], [-4., 4.]]], dtype=np.float64)
        assert np.array_equal(res_vec, exp_vec)
        # uniform no keep size
        res_vec = MatVector.eval_func(np.sum, data_vec, True, False)
        exp_vec = np.array([10, -10])
        assert np.array_equal(res_vec, exp_vec)
        # not uniform
        res_vec = MatVector.eval_func(mat_min, data_vec, False)
        exp_vec = list([np.array([[-1., -2.], [-3., -4.]]), np.array([[1., 2.], [3., 4.]])])
        assert list_equal(res_vec, exp_vec)

    def test_from_expression_single_no_const(self):
        # testing for a single-sized expression
        exp = "[[sin(t)]]"
        # single-element array
        t = np.array([0])
        exp_arr = np.array([[[0]]], dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # int
        t = 0
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # float
        t = 0.
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # vector
        t = np.array([0, np.pi / 2, np.pi])
        exp_arr = np.array([[[0., 1., 0.]]], dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.allclose(res_arr, exp_arr)

    def test_from_expression_single_const(self):
        # testing for a single-sized expression
        exp = "[[1.]]"
        # single-element array
        t = np.array([0])
        exp_arr = np.array([[[1]]], dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # int
        t = 0
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # float
        t = 0.
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # vector
        t = np.array([0, np.pi / 2, np.pi])
        exp_arr = np.array([[[1., 1., 1.]]], dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)

    def test_from_expression_brackets(self):
        # testing for a non-bracket expression
        exp = "sin(t)"
        # single-element array
        t = np.array([0])
        exp_arr = np.array([[[0]]], dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        exp = "[sin(t)]"
        exp_arr = np.array([[[0]]], dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)

    def test_from_expression_matrix_no_const(self):
        # testing for a matrix expression
        exp = "[[sin(t), cos(t)], [-cos(t), sin(t)]]"
        # single-element array
        t = np.array([0])
        exp_arr = np.array([[[0.],
                             [1.]],
                            [[-1.],
                             [0.]]],
                           dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # int
        t = 0
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # float
        t = 0.
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # vector
        t = np.array([0, np.pi / 2, np.pi])
        exp_arr = np.array([[[0., 1., 0],
                             [1., 0., -1.]],
                            [[-1., 0., 1.],
                             [0., 1., 0.]]],
                           dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.allclose(res_arr, exp_arr)

    def test_from_expression_matrix_some_const(self):
        # testing for a matrix expression with some constants in it
        exp = "[[sin(t), cos(t)], [-cos(t), 2]]"
        # single-element array
        t = np.array([0])
        exp_arr = np.array([[[0.],
                             [1.]],
                            [[-1.],
                             [2.]]],
                           dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # int
        t = 0
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # float
        t = 0.
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # vector
        t = np.array([0, np.pi / 2, np.pi])
        exp_arr = np.array([[[0., 1., 0],
                             [1., 0., -1.]],
                            [[-1., 0., 1.],
                             [2., 2., 2.]]],
                           dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.allclose(res_arr, exp_arr)

    def test_from_expression_matrix_all_const(self):
        # testing for a matrix expression with all constants
        exp = "[[1, 2], [3, 4], [5, 6]]"
        # single-element array
        t = np.array([0])
        exp_arr = np.array([[[1.],
                             [2.]],
                            [[3.],
                             [4.]],
                            [[5.],
                             [6.]]],
                           dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # int
        t = 0
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # float
        t = 0.
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)
        # vector
        t = np.array([0, np.pi / 2, np.pi])
        exp_arr = np.array([[[1., 1., 1],
                             [2., 2., 2.]],
                            [[3., 3., 3.],
                             [4., 4., 4.]],
                            [[5., 5., 5.],
                             [6., 6., 6.]]],
                           dtype=float)
        res_arr = MatVector.from_expression(exp, t)
        assert np.array_equal(res_arr, exp_arr)

    def test_r_multiply_simple(self):
        __MAX_TOL = 1e-11
        a_mat = np.random.rand(2, 2, 1)
        b_mat = np.random.rand(2, 2, 1)
        c_mat = np.random.rand(2, 2, 1)
        res_mat = MatVector.r_multiply(a_mat, b_mat, c_mat)
        exp_mat = np.zeros((2, 2, 1))
        exp_mat[:, :, 0] = a_mat[:, :, 0] @ b_mat[:, :, 0] @ c_mat[:, :, 0]
        assert np.allclose(exp_mat - res_mat, np.zeros((2, 2, 1)), atol=__MAX_TOL)

    def test_r_multiply(self):
        __CALC_PRECISION__ = 1e-5

        def check(l_inp: np.ndarray, r_inp: np.ndarray):
            res = l_inp - r_inp
            assert (np.max(np.abs(res).flatten()) < __CALC_PRECISION__)

        loaded_data = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matvector_data.mat'))
        a_arr = loaded_data['aArray']
        b_mat = a_arr[:, :, 0]
        b_arr = a_arr[:, :, 1:]
        res_mat = MatVector.r_multiply(a_arr, a_arr)
        exp_mat = np.zeros(a_arr.shape)
        for i in range(a_arr.shape[2]):
            exp_mat[:, :, i] = a_arr[:, :, i] @ a_arr[:, :, i]
        check(res_mat, exp_mat)
        res_mat = MatVector.r_multiply(a_arr, b_mat)
        for i in range(a_arr.shape[2]):
            exp_mat[:, :, i] = a_arr[:, :, i] @ b_mat
        check(res_mat, exp_mat)
        with pytest.raises(Exception) as e:
            _ = MatVector.r_multiply(a_arr, b_arr)
        assert 'wrongInput:Incorrect size of b_arr' in str(e.value)

    def test_compare_mat_vector_multiply(self):
        __CALC_PRECISION__ = 1e-5

        def check(l_inp: np.ndarray, r_inp: np.ndarray):
            res = l_inp - r_inp
            assert (np.max(np.abs(res).flatten()) < __CALC_PRECISION__)

        loaded_data = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matvector_data.mat'))
        a_arr = loaded_data['aArray']
        b_mat = a_arr[1, :, :].squeeze()
        c_arr = MatVector.r_multiply(a_arr, a_arr, use_sparse_matrix=False)
        d_arr = MatVector.r_multiply(a_arr, a_arr, use_sparse_matrix=True)
        check(c_arr, d_arr)
        c_arr = MatVector.r_multiply(a_arr[0:5, 0:6, :], a_arr[0:6, 0:7, :], a_arr[0:7, 0:8, :],
                                     use_sparse_matrix=False)
        d_arr = MatVector.r_multiply(a_arr[0:5, 0:6, :], a_arr[0:6, 0:7, :], a_arr[0:7, 0:8, :],
                                     use_sparse_matrix=True)
        check(c_arr, d_arr)
        c_arr = MatVector.r_multiply(a_arr, b_mat, use_sparse_matrix=False)
        d_arr = MatVector.r_multiply(a_arr, b_mat, use_sparse_matrix=True)
        check(c_arr, d_arr)
        c_arr = MatVector.r_multiply_by_vec(a_arr[0:7, 0:10, 0:100], b_mat[0:10, 0:100], False)
        d_arr = MatVector.r_multiply_by_vec(a_arr[0:7, 0:10, 0:100], b_mat[0:10, 0:100], True)
        check(c_arr, d_arr)

        with pytest.raises(Exception) as e:
            b_arr = np.array([1, 2]).T
            _ = MatVector.r_multiply_by_vec(a_arr, b_arr, False)
            _ = MatVector.r_multiply_by_vec(a_arr, b_arr, True)
        assert "wrongInput:b_mat is expected to be 2-dimensional array" in str(e.value)

    def test_sort_rows_tol(self):

        def check_int(res_mat: np.ndarray, checkint_inp_mat: np.ndarray, ind_vec: np.ndarray, ind_sort_vec: np.ndarray):
            assert np.allclose(res_mat, checkint_inp_mat[ind_vec.flatten()])
            assert np.allclose(ind_sort_vec, ind_vec.flatten())

        def check(ind_vec: np.ndarray, tol: float, check_inp_mat: np.ndarray):
            res_mat, ind_sort_vec, _ = sort_rows_tol(check_inp_mat, tol)
            check_int(res_mat, check_inp_mat, ind_vec, ind_sort_vec)
            res_mat, ind_sort_vec, ind_rev_sort_vec = sort_rows_tol(check_inp_mat, tol)
            check_int(res_mat, check_inp_mat, ind_vec, ind_sort_vec)
            assert np.allclose(res_mat[ind_rev_sort_vec, :], check_inp_mat)

        inp_mat = np.array([[1, 2], [1 + 1e-14, 1]], dtype=np.float64)
        check(np.array([0, 1]), 1e-16, inp_mat)
        check(np.array([1, 0]), 1e-14, inp_mat)

        inp_mat = np.array([[1, 2], [1 + 1e-14, 1], [1 - 1e-14, 0]], dtype=np.float64)

        check(np.array([2, 1, 0]), 1e-13, inp_mat)
        check(np.array([2, 0, 1]), 1e-15, inp_mat)

    def test_r_svd_multyply_by_vec(self):
        __MAX_TOL = 1e-10
        a_mat, b_mat, c_vec = TestGen.__aux_symmetric_mat_vector_arrays()
        res_vec = SymmetricMatVector.r_svd_multiply_by_vec(a_mat, c_vec)
        #
        TestGen.__aux_symmetric_mat_vector_check(lambda x: x, a_mat, c_vec, res_vec, False, 2, __MAX_TOL)

    def test_lr_svd_multiply(self):
        __MAX_TOL = 1e-10
        a_mat, b_mat, c_vec = TestGen.__aux_symmetric_mat_vector_arrays()
        res_vec = SymmetricMatVector.lr_svd_multiply(a_mat, b_mat)
        #
        TestGen.__aux_symmetric_mat_vector_check(lambda x: x, a_mat, b_mat, res_vec, True, 3, __MAX_TOL)

    def test_lr_svd_multiply_by_vec(self):
        __MAX_TOL = 1e-10
        a_mat, b_mat, c_vec = TestGen.__aux_symmetric_mat_vector_arrays()
        res_vec = SymmetricMatVector.lr_svd_multiply_by_vec(a_mat, c_vec)
        #
        TestGen.__aux_symmetric_mat_vector_check(lambda x: x, a_mat, c_vec, res_vec, True, 2, __MAX_TOL)

    def test_lr_svd_divide_vec(self):
        __MAX_TOL = 1e-10
        a_mat, b_mat, c_vec = TestGen.__aux_symmetric_mat_vector_arrays()
        res_vec = SymmetricMatVector.lr_svd_divide_vec(a_mat, c_vec)
        #
        TestGen.__aux_symmetric_mat_vector_check(lambda x: la.inv(x), a_mat, c_vec, res_vec, True, 1, __MAX_TOL)

    @staticmethod
    def __aux_symmetric_mat_vector_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a_mat = np.zeros((2, 2, 2))
        a_mat[:, :, 0] = ([[0, 1], [1, 0]])
        a_mat[:, :, 1] = ([[5, 2], [2, 1]])
        #
        sup_mat = np.array([[1], [0]])
        c_vec = np.tile(sup_mat, [1, 2])
        #
        b_mat = np.zeros((2, 3, 2))
        b_mat[:, :, 0] = ([[4, 6, 1], [-6, 2, 4]])
        b_mat[:, :, 1] = ([[8, 3, 8], [4, 3, 7]])
        return a_mat, b_mat, c_vec

    @staticmethod
    def __aux_symmetric_mat_vector_check(fo_func, inp1_array, inp2_array, res_array, is_lr_op, n_out_dims, accuracy):
        size_vec = np.shape(res_array)
        n_points = size_vec[-1]
        out_vec = np.zeros(size_vec, dtype=np.float64)
        for i in range(n_points):
            s_mat, u_mat = la.eigh(inp1_array[:, :, i])
            s_mat = np.diag(s_mat)
            if (n_out_dims == 1) or (n_out_dims == 2):
                arg_2_mat = inp2_array[:, i]
            else:
                arg_2_mat = inp2_array[:, :, i]
            #
            if is_lr_op:
                extra_mat = u_mat @ arg_2_mat
                res_mat = extra_mat.T @ fo_func(s_mat) @ extra_mat
            else:
                res_mat = u_mat.T @ fo_func(s_mat) @ u_mat @ arg_2_mat
            #
            if n_out_dims == 1:
                out_vec[i] = res_mat
            elif n_out_dims == 2:
                out_vec[:, i] = res_mat
            else:
                out_vec[:, :, i] = res_mat
        res = out_vec - res_array
        assert (np.max(np.abs(res).flatten()) < accuracy)

    def test_square_mat_vect_inv_3d(self):
        load_data = TestGen.__SQUARE_MAT_VEC_DATA['inv']
        data_arr = load_data[0][0]['input'][0][0][0][0]
        out_arr_true = load_data[0][0]['output'][0][0][0][0]
        out_arr = SquareMatVector.inv(data_arr)
        assert np.array_equal(out_arr, out_arr_true)

    def test_square_mat_vect_sqrtm_pos_3d(self):
        load_data = TestGen.__SQUARE_MAT_VEC_DATA['sqrtm_pos']
        data_arr = load_data[0][0]['input'][0][0][0][0]
        out_arr_true = load_data[0][0]['output'][0][0][0][0]
        out_arr = SquareMatVector.sqrtm_pos(data_arr)
        assert np.array_equal(out_arr, out_arr_true)

    def test_square_mat_vect_make_pos_definite_or_nan_3d(self):
        load_data = TestGen.__SQUARE_MAT_VEC_DATA['make_pos_definite_or_nan']
        data_arr = load_data[0][0]['input'][0][0][0][0]
        out_arr_true = load_data[0][0]['output'][0][0][0][0][:, :, 1]
        out_arr = SquareMatVector.make_pos_definite_or_nan(data_arr)
        assert np.array_equal(out_arr[:, :, 1], out_arr_true)
        assert np.isnan(out_arr[:, :, 0]).all()

    def test_square_mat_vect_make_pos_definite_by_eig_3d(self):
        load_data = TestGen.__SQUARE_MAT_VEC_DATA['make_pos_definite_by_eig']
        data_arr = load_data[0][0]['input'][0][0][0][0]
        out_arr_true = load_data[0][0]['output'][0][0][0][0]
        out_arr = SquareMatVector.make_pos_definite_by_eig(data_arr)
        assert np.isclose(out_arr, out_arr_true).all()

    def test_square_mat_vect_lr_multiply(self):
        load_data = TestGen.__SQUARE_MAT_VEC_DATA['lr_multiply']
        for i in range(len(load_data[0][0]['input'][0][0][0])):
            flag = load_data[0][0]['input'][0][0][0][i][0][2][0]
            inp_a_arr = load_data[0][0]['input'][0][0][0][i][0][0]
            inp_b_arr = load_data[0][0]['input'][0][0][0][i][0][1]
            out_arr_true = load_data[0][0]['output'][0][0][0][i]
            out_arr = SquareMatVector.lr_multiply(inp_b_arr, inp_a_arr, flag)
            assert np.array_equal(out_arr, out_arr_true)

    def test_square_mat_vect_lr_multiply_by_vec(self):
        load_data = TestGen.__SQUARE_MAT_VEC_DATA['lr_multiply_by_vec']
        inp_a_arr = load_data[0][0]['input'][0][0][0][0]
        inp_b_arr = load_data[0][0]['input'][0][0][0][1]
        out_arr_true = load_data[0][0]['output'][0][0][0][0]
        out_arr = SquareMatVector.lr_multiply_by_vec(inp_b_arr, inp_a_arr)
        assert np.array_equal(out_arr, out_arr_true)

    def test_square_mat_vect_lr_divide_vec(self):
        load_data = TestGen.__SQUARE_MAT_VEC_DATA['lr_divide_vec']
        inp_a_arr = load_data[0][0]['input'][0][0][0][0]
        inp_b_arr = load_data[0][0]['input'][0][0][0][1]
        out_arr_true = load_data[0][0]['output'][0][0][0][0]
        out_arr = SquareMatVector.lr_divide_vec(inp_b_arr, inp_a_arr)
        assert np.isclose(out_arr, out_arr_true).all()
