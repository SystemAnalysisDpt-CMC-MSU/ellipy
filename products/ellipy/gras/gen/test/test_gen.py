from ellipy.gras.gen.gen import *
from typing import List
import numpy as np
import scipy.io
import os
import pytest


class TestGen:
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
        b_mat = a_arr[:, :, 1]
        b_arr = a_arr[:, :, 2:]
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
            b_mat = np.array([[[1, 2, 3, 0], [3, 4, 5, 0]], [[5, 6, 7, 0], [7, 8, 9, 0]]], dtype=np.float64)
            _ = MatVector.r_multiply(a_arr, b_mat)
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
        c_arr = MatVector.r_multiply(a_arr[1:5,1:6,:], a_arr[1:6,1:7,:], a_arr[1:7,1:8,:],  use_sparse_matrix=False)
        d_arr = MatVector.r_multiply(a_arr[1:5,1:6,:], a_arr[1:6,1:7,:], a_arr[1:7,1:8,:],  use_sparse_matrix=True)
        check(c_arr, d_arr)
        c_arr = MatVector.r_multiply(a_arr, b_mat, use_sparse_matrix=False)
        d_arr = MatVector.r_multiply(a_arr, b_mat, use_sparse_matrix=True)
        check(c_arr, d_arr)
        c_arr = MatVector.r_multiply_by_vec(a_arr[1:7, 1:10, 1:100], b_mat[1:10, 1:100], False)
        d_arr = MatVector.r_multiply_by_vec(a_arr[1:7, 1:10, 1:100], b_mat[1:10, 1:100], True)
        check(c_arr, d_arr)
        assert np.allclose(c_arr, d_arr)
        with pytest.raises(Exception) as e:
            b_arr = np.array([1, 2]).T
            _ = MatVector.r_multiply_by_vec(a_arr, b_arr, False)
            _ = MatVector.r_multiply_by_vec(a_arr, b_arr, True)
        assert "wrongInput:b_mat is expected to be 2-dimensional array" in str(e.value)
