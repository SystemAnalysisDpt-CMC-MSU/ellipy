from ellipy.elltool.core.core import *
from ellipy.elltool.core.hyperplane.Hyperplane import *
from ellipy.elltool.conf.properties.Properties import *
import pytest
import numpy as np


class TestEllTCMultiDim:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def ell_unitball(self, *args, **kwargs):
        return ell_unitball(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def hyperplane(self, *args, **kwargs):
        return Hyperplane(*args, **kwargs)

    def test_max_eig(self):
        _check_maxeig_ans_mineig(self, True)

    def test_min_eig(self):
        _check_maxeig_ans_mineig(self, False)

    def test_trace(self):
        def test_correct(flag):
            if (flag == 2) or (flag == 6) or (flag == 16):
                test_ell_array, ans_num_array, *_ = self.__create_typical_array(flag)
            else:
                test_ell_array, *_, ans_num_array = self.__create_typical_array(flag)
            if test_ell_array.size > 0:
                test_num_array = test_ell_array.flat[0].trace(test_ell_array)
            else:
                test_num_array = self.ellipsoid().trace(test_ell_array)
            assert np.array_equal(ans_num_array, test_num_array)
            return ans_num_array, test_num_array

        def test_error(flag):
            test_ell_array, _, error_str, *_ = self.__create_typical_array(flag)
            if flag == 1:
                with pytest.raises(Exception) as e:
                    if test_ell_array.size > 0:
                        test_ell_array.flat[0].trace(test_ell_array)
                    else:
                        self.ellipsoid().trace(test_ell_array)
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    if test_ell_array.size > 0:
                        test_ell_array.flat[0].trace(test_ell_array)
                    else:
                        self.ellipsoid().trace(test_ell_array)
                assert error_str in str(e.value)

        test_correct(6)
        test_correct(2)
        test_correct(7)
        test_correct(8)
        ans_num_array_res, test_num_array_res = test_correct(16)
        assert type(ans_num_array_res) == type(test_num_array_res)

        test_error(1)
        test_error(14)
        test_error(15)

    def test_volume(self):
        def test_correct(flag):
            if flag == 16:
                test_ell_array, out_ans_double_array, *_ = self.__create_typical_array(flag)
            elif flag == 2:
                test_ell_array, _, out_ans_double_array, _ = self.__create_typical_array(flag)
            elif flag == 3:
                test_ell_array, _, out_ans_double_array = self.__create_typical_array(flag)
            else:
                test_ell_array, *_, out_ans_double_array = self.__create_typical_array(flag)
            if test_ell_array.size > 0:
                out_test_double_array = test_ell_array.flat[0].volume(test_ell_array)
            else:
                out_test_double_array = self.ellipsoid().volume(test_ell_array)
            assert np.array_equal(out_ans_double_array, out_test_double_array.flatten())
            return out_ans_double_array, out_test_double_array

        def test_error(flag):
            test_ell_array, _, error_str, *_ = self.__create_typical_array(flag)
            if flag == 1:
                with pytest.raises(Exception) as e:
                    if test_ell_array.size > 0:
                        test_ell_array.flat[0].volume(test_ell_array)
                    else:
                        self.ellipsoid().volume(test_ell_array)
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    if test_ell_array.size > 0:
                        test_ell_array.flat[0].volume(test_ell_array)
                    else:
                        self.ellipsoid().volume(test_ell_array)
                assert error_str in str(e.value)

        # Check degenerate self.ellipsoid
        test_correct(4)
        # Check dim = 1 with two different centers
        test_correct(2)
        test_correct(3)
        ans_double_array, test_double_array = test_correct(16)
        assert type(ans_double_array) == type(test_double_array)
        # Empty self.ellipsoid
        test_error(1)
        test_error(14)
        test_error(15)

    def test_is_degenerate(self):
        def test_correct(inp_test_ell_array, inp_is_ans_array):
            if inp_test_ell_array.size > 0:
                out_is_test_res = inp_test_ell_array.flat[0].is_degenerate(inp_test_ell_array)
            else:
                out_is_test_res = self.ellipsoid().is_degenerate(inp_test_ell_array)
            assert np.array_equal(inp_is_ans_array, out_is_test_res)

            return out_is_test_res

        def test_error(flag):
            test_ell_array_res, *_, error_str = self.__create_typical_array(flag)
            test_ell_array_res = np.array(test_ell_array_res)
            if flag == 1:
                with pytest.raises(Exception) as e:
                    test_ell_array_res.flat[0].is_degenerate(test_ell_array_res[0])
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    test_ell_array_res.flat[0].is_degenerate(np.array([test_ell_array_res]))
                assert error_str[0] in str(e.value)

        # Not degerate self.ellipsoid
        test_ell_array, is_ans_array = self.__create_typical_array(5)
        test_correct(np.array(test_ell_array), np.array(is_ans_array))
        # Degenerate self.ellipsoid
        array_size_vec = np.array([2, 1, 1, 1, 3, 1, 1])
        test_ell_array = _create_object_array(array_size_vec, self.ellipsoid,
                                              np.diag(np.array([1, 2, 3, 4, 0])),
                                              np.array([[1]]), 1)
        is_ans_array = _create_object_array(array_size_vec, np.ones, np.array([1]), np.array([1]), 1)
        test_correct(test_ell_array, is_ans_array)
        array_size_vec = np.array([1, 1, 2, 3, 1, 2, 1])
        diag_mat = np.zeros((100, 100))
        np.fill_diagonal(diag_mat[50:100, 50:100], 1)
        test_ell_array = _create_object_array(array_size_vec, self.ellipsoid, diag_mat, np.array([1]), 1)
        is_ans_array = _create_object_array(array_size_vec, np.ones, np.array([1]), np.array([1]), 1)
        is_test_res = test_correct(test_ell_array, is_ans_array)
        assert type(is_ans_array) == type(is_test_res)
        test_ell_array, _, is_ans_array = self.__create_typical_array(16)
        is_test_res = test_correct(test_ell_array, is_ans_array)
        assert type(is_ans_array) == type(is_test_res)
        # Empty self.ellipsoid
        test_error(1)
        test_error(14)
        test_error(15)

    def __create_typical_array(self, *args):
        return _create_typical_array(self, *args)

    def test_is_empty(self):
        def test_correct(inp_test_ell_array, inp_is_ans_array):
            if inp_test_ell_array.size > 0:
                out_is_test_res = inp_test_ell_array.flat[0].is_empty(inp_test_ell_array).flatten()
            else:
                out_is_test_res = self.ellipsoid().is_empty(inp_test_ell_array).flatten()
            assert np.array_equal(out_is_test_res, inp_is_ans_array)
            return out_is_test_res

        array_size_vec = np.array([2, 1, 1, 1, 1, 3, 1, 1])
        test_ell_array = _create_object_array(array_size_vec, self.ellipsoid, 1, 1, 0).flatten()
        is_ans_array = _create_object_array(array_size_vec, np.ones, np.array([1]), np.array([1]), 1).flatten()
        test_correct(test_ell_array, is_ans_array)

        # Check not empty self.ellipsoid
        test_ell_array, is_ans_array = self.__create_typical_array(5)
        test_correct(test_ell_array, is_ans_array)

        array_size_vec = np.array([1, 1, 1, 1, 1, 4, 1, 1, 3])
        diag_mat = np.zeros((100, 100))
        np.fill_diagonal(diag_mat[50:100, 50:100], 1)
        test_ell_array = _create_object_array(array_size_vec, self.ellipsoid, diag_mat, np.array([1]), 1)
        is_ans_array = _create_object_array(array_size_vec, np.zeros, np.array([1]), np.array([1]), 1).flatten()
        test_correct(test_ell_array, is_ans_array)

        test_ell_array, *_, is_ans_array = self.__create_typical_array(16)
        is_test_res = test_correct(test_ell_array, is_ans_array)
        assert type(is_ans_array) == type(is_test_res)


def _create_typical_array(ell_factory_obj, flag):
    array_size_vec = np.array([2, 1, 1, 2, 1, 3, 1])
    if flag == 1:
        array_size_vec = np.array([2, 1, 3, 2, 1, 1, 4])
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid, 1, 1, 0)
        ans_num_array = _create_object_array(array_size_vec, np.diag, np.array([0]), np.array([1]), 1).flatten()
        is_ans_array = np.ones(array_size_vec).flatten()
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array, ans_num_array, is_ans_array, error_str
    if flag == 2:
        array_size_vec = np.array([1, 2, 4, 3, 2, 1])
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball, 1, 1, 1).flatten()
        ans_num_array = _create_object_array(array_size_vec, np.diag, np.array([1]), np.array([1]), 1).flatten()
        ans_volume_double_array = _create_object_array(array_size_vec, np.diag, np.array([2]),
                                                       np.array([1]), 1).flatten()
        is_ans_array = np.ones(array_size_vec).flatten()
        return test_ell_array, ans_num_array, ans_volume_double_array, is_ans_array
    if flag == 3:
        array_size_vec = np.array([1, 1, 1, 1, 1, 7, 1, 1, 7])
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            np.eye(5, 5), np.array([1]), 1).flatten()
        ans_num_array = _create_object_array(array_size_vec, np.diag, np.array([5]), np.array([1]), 1).flatten()
        volume_double = 8 * (np.pi**2)/15
        ans_volume_double_array = _create_object_array(array_size_vec, np.diag,
                                                       np.array([volume_double]), np.array([1]), 1).flatten()
        return test_ell_array, ans_num_array, ans_volume_double_array
    if flag == 4:
        array_size_vec = np.array([2, 1, 3, 2, 1, 1, 4, 1, 1])
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            np.diag(np.array([1, 2, 3, 4, 0])), np.array([1]), 1).flatten()
        ans_dim_num_array = _create_object_array(array_size_vec, np.diag, np.array([5]), np.array([1]), 1).flatten()
        ans_rank_num_array = _create_object_array(array_size_vec, np.diag, np.array([4]), np.array([1]), 1).flatten()
        ans_volume_double_array = _create_object_array(array_size_vec, np.diag,
                                                       np.array([0]), np.array([1]), 1).flatten()
        return test_ell_array, ans_dim_num_array, ans_rank_num_array, ans_volume_double_array
    if flag == 5:
        array_size_vec = np.array([1, 2, 4, 3, 2])
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball,
            1, 1, 1).flatten()
        is_ans_array = _create_object_array(array_size_vec, np.zeros, np.array([1]), np.array([1]), 1).flatten()
        return test_ell_array, is_ans_array
    if flag == 6:
        array_size_vec = np.array([1, 1, 2, 3, 2, 1, 1, 1, 4])
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            np.zeros((100, 100)), np.eye(1), 1).flatten()
        ans_num_array = _create_object_array(array_size_vec, np.diag, np.array([0]), np.eye(1), 1).flatten()
        return test_ell_array, ans_num_array
    if flag == 7:
        array_size_vec = np.array([2, 3, 2, 1, 1, 1, 4, 1, 1])
        my_mat = np.diag(np.linspace(0.0, 100.0, 101))
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            my_mat, np.array([1]), 1).flatten()
        ans_max_num_array = _create_object_array(array_size_vec, np.diag, np.array([100]),
                                                 np.array([1]), 1).flatten()
        ans_min_num_array = _create_object_array(array_size_vec, np.diag, np.array([0]), np.array([1]), 1).flatten()
        ans_trace_num_array = _create_object_array(array_size_vec, np.diag,
                                                   np.array([np.sum(np.linspace(0.0, 100.0, 101))]),
                                                   np.array([1]), 1).flatten()
        return test_ell_array, ans_max_num_array, ans_min_num_array, ans_trace_num_array
    if flag == 8:
        array_size_vec = np.array([1, 1, 1, 1, 1, 7, 1, 1, 7])
        my_mat = np.random.rand(10, 10)
        my_mat = my_mat @ my_mat.T
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            my_mat, np.array([1]), 1)
        ans_max_num_array = _create_object_array(array_size_vec, np.diag,
                                                 np.array([np.max(np.linalg.eigvalsh(my_mat))]),
                                                 np.array([1]), 1).flatten()
        ans_min_num_array = _create_object_array(array_size_vec, np.diag,
                                                 np.array([np.min(np.linalg.eigvalsh(my_mat))]),
                                                 np.array([1]), 1).flatten()
        ans_trace_num_array = _create_object_array(array_size_vec, np.diag, np.array([np.trace(my_mat)]),
                                                   np.array([1]), 1).flatten()
        return test_ell_array, ans_max_num_array, ans_min_num_array, ans_trace_num_array
    if flag == 9:
        __MAX_TOL = Properties.get_rel_tol()
        array_size_vec = np.array([1, 1, 1, 1, 1, 7, 1, 1, 7])
        my_1_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball, 2, 1, 1).flatten()
        my_2_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            np.diag([1 + __MAX_TOL, 1 + __MAX_TOL]), np.array([1]), 1).flatten()
        is_ans_array = np.ones(array_size_vec).flatten()
        return my_1_ell_array, my_2_ell_array, is_ans_array
    if flag == 10:
        __MAX_TOL = Properties.get_rel_tol()
        array_size_vec = np.array([1, 1, 2, 1, 1, 1, 2, 1, 1])
        my_1_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball,
            5, 1, 1).flatten()
        my_2_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            np.diag(np.tile(([1 + 100 * __MAX_TOL]), (1, 5)).flatten()),
            np.array([1]), 1).flatten()
        is_ans_array = np.zeros(array_size_vec).flatten()
        return my_1_ell_array, my_2_ell_array, is_ans_array
    if flag == 11:
        array_size_vec = np.array([1, 1, 3, 1, 1, 1, 2, 1, 1])
        my_1_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball,
            5, 1, 1).flatten()
        my_2_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball, 4, 1, 1).flatten()
        is_ans_array = np.zeros(array_size_vec).flatten()
        report_str = 'wrongInput:emptyEllipsoid'
        return my_1_ell_array, my_2_ell_array, is_ans_array, report_str
    if flag == 12:
        __MAX_TOL = Properties.get_rel_tol()
        array_size_vec = np.array([1, 1, 2, 1, 1, 1, 1, 1, 2])
        my_1_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball,
            10, 1, 1).flatten()
        my_2_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball,
            (2 * __MAX_TOL) * np.ones((10, 1)), np.eye(10, 10), 1).flatten()
        is_ans_array = np.zeros(array_size_vec).flatten()
        report_str = 'wrongInput:emptyEllipsoid'
        return my_1_ell_array, my_2_ell_array, is_ans_array, report_str
    if flag == 13:
        test_ell_array = np.empty([1, 0, 0, 1, 5], dtype=object)
        test_2_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball,
            np.array([3]), np.array([1]), 1)
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array, test_2_ell_array, error_str
    if flag == 14:
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball,
            3, 1, 1)
        test_ell_array[1, 0, 0, 1, 0, 2, 0] = ell_factory_obj.ellipsoid()
        test_2_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ell_unitball,
            3, 1, 1).flatten()
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array.flatten(), test_2_ell_array, error_str
    if flag == 15:
        test_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            np.array([[3]]), np.array([[1]]), 1)
        test_ell_array[1, 0, 0, 1, 0, 2, 0] = ell_factory_obj.ellipsoid()
        test_2_ell_array = _create_object_array(
            array_size_vec, ell_factory_obj.ellipsoid,
            np.array([[3]]), np.array([[1]]), 1).flatten()
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array.flatten(), test_2_ell_array, error_str
    if flag == 16:
        array_size_vec = np.array([1, 0, 0, 1, 5])
        test_ell_array = np.empty(array_size_vec, dtype=object).flatten()
        ans_double_array = np.zeros(array_size_vec).flatten()
        is_ans_array = np.ones(array_size_vec).flatten()
        return test_ell_array, ans_double_array, is_ans_array


def _create_object_array(array_size_vec, func, first_arg, second_arg, n_arg) -> np.ndarray:
    n_elems = np.prod(array_size_vec).flat[0]
    object_list = None
    if n_arg == 0:
        object_list = [func() for _ in range(n_elems)]
    elif n_arg == 1:
        object_list = [func(first_arg) for _ in range(n_elems)]
    elif n_arg == 2:
        object_list = [[func(first_arg), func(second_arg)] for _ in range(n_elems)]
    return np.reshape(np.array(object_list).flatten(), array_size_vec)


def _check_maxeig_ans_mineig(ell_factory_obj, is_maxeig_check):
    def test_correct(flag):
        if is_maxeig_check:
            test_ell_array, out_ans_num_array, *_ = _create_typical_array(ell_factory_obj, flag)
            if test_ell_array.size > 0:
                out_test_num_array = test_ell_array.flat[0].max_eig(test_ell_array)
            else:
                out_test_num_array = ell_factory_obj.ellipsoid().max_eig(test_ell_array)
        else:
            if (flag == 2) or (flag == 6) or (flag == 16):
                test_ell_array, out_ans_num_array, *_ = _create_typical_array(ell_factory_obj, flag)
            else:
                test_ell_array, _, out_ans_num_array, *_ = _create_typical_array(ell_factory_obj, flag)
            if test_ell_array.size > 0:
                out_test_num_array = test_ell_array.flat[0].min_eig(test_ell_array)
            else:
                out_test_num_array = ell_factory_obj.ellipsoid().min_eig(test_ell_array)
        assert np.array_equal(out_ans_num_array, out_test_num_array)
        return out_ans_num_array, out_test_num_array

    def test_error(flag):
        test_ell_array, _, error_str, *_ = _create_typical_array(ell_factory_obj, flag)
        if is_maxeig_check:
            if flag == 1:
                with pytest.raises(Exception) as e:
                    [x.max_eig(x) for x in test_ell_array.flatten()]
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    if test_ell_array.size > 0:
                        test_ell_array.flat[0].max_eig(test_ell_array)
                    else:
                        ell_factory_obj.ellipsoid().max_eig(test_ell_array)
                assert error_str in str(e.value)
        else:
            if flag == 1:
                with pytest.raises(Exception) as e:
                    if test_ell_array.size > 0:
                        test_ell_array.flat[0].min_eig(test_ell_array)
                    else:
                        ell_factory_obj.ellipsoid().min_eig(test_ell_array)
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    if test_ell_array.size > 0:
                        test_ell_array.flat[0].min_eig(test_ell_array)
                    else:
                        ell_factory_obj.ellipsoid().min_eig(test_ell_array)
                assert error_str in str(e.value)

    # Check degenerate matrix

    test_correct(6)
    test_correct(2)
    test_correct(7)
    test_correct(8)
    ans_num_array, test_num_array = test_correct(16)
    assert type(test_num_array) == type(ans_num_array)

    test_error(1)
    test_error(14)
    test_error(15)
