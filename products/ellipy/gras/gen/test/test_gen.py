from ellipy.gras.gen.gen import *
import numpy as np


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
