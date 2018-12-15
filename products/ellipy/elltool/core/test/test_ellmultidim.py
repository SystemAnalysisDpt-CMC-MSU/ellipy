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
        check_maxeig_ans_mineig(True)

    def test_min_eig(self):
        check_maxeig_ans_mineig(False)

    def test_trace(self):
        def __test_correct(flag):
            if (flag == 2) or (flag == 6) or (flag == 16):
                test_ell_array, ans_num_array, *_ = create_typical_array(self, flag)
            else:
                test_ell_array, *_, ans_num_array = create_typical_array(self, flag)
            test_num_array = AEllipsoid.trace(test_ell_array)
            assert np.array_equal(ans_num_array, test_num_array)
            return ans_num_array, test_num_array

        def __test_error(flag):
            test_ell_array, _, error_str, *_ = create_typical_array(self, flag)
            if flag == 1:
                with pytest.raises(Exception) as e:
                    AEllipsoid.trace(test_ell_array)
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    AEllipsoid.trace(test_ell_array)
                assert error_str in str(e.value)

        __test_correct(6)
        __test_correct(2)
        __test_correct(7)
        __test_correct(8)
        __ans_num_array_res, __test_num_array_res = __test_correct(16)
        assert type(__ans_num_array_res) == type(__test_num_array_res)

        __test_error(1)
        __test_error(14)
        __test_error(15)

    def test_volume(self):
        def __test_correct(flag):
            if flag == 16:
                test_ell_array, ans_double_array, *_ = create_typical_array(self, flag)
            elif flag == 2:
                test_ell_array, _, ans_double_array, _ = create_typical_array(self, flag)
            elif flag == 3:
                test_ell_array, _, ans_double_array = create_typical_array(self, flag)
            else:
                test_ell_array, *_, ans_double_array = create_typical_array(self, flag)
            test_double_array = AEllipsoid.volume(test_ell_array)
            assert np.array_equal(ans_double_array, test_double_array.flatten())
            return ans_double_array, test_double_array

        def __test_error(flag):
            test_ell_array, _, error_str, *_ = create_typical_array(self, flag)
            if flag == 1:
                with pytest.raises(Exception) as e:
                    AEllipsoid.volume(test_ell_array)
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    AEllipsoid.volume(test_ell_array)
                assert error_str in str(e.value)

        # Check degenerate self.ellipsoid
        __test_correct(4)
        # Check dim = 1 with two different centers
        __test_correct(2)
        __test_correct(3)
        __ans_double_array, __test_double_array = __test_correct(16)
        assert type(__ans_double_array) == type(__test_double_array)
        # Empty self.ellipsoid
        __test_error(1)
        __test_error(14)
        __test_error(15)

    def test_is_degenerate(self):
        def __test_correct(test_ell_array, is_ans_array):
            is_test_res = AEllipsoid.is_degenerate(test_ell_array)
            assert np.array_equal(is_ans_array, is_test_res)

            return is_test_res

        def __test_error(flag):
            test_ell_array_res, *_, error_str = self.__create_typical_array(flag)
            if flag == 1:
                with pytest.raises(Exception) as e:
                    AEllipsoid.is_degenerate(test_ell_array_res[0])
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    AEllipsoid.is_degenerate(np.array([test_ell_array_res]))
                assert error_str[0] in str(e.value)

        # Not degerate self.ellipsoid
        __test_ell_array, __is_ans_array = self.__create_typical_array(5)
        __test_correct(__test_ell_array, __is_ans_array)
        # Degenerate self.ellipsoid
        __array_size_vec = np.array([2, 1, 1, 1, 3, 1, 1])
        __test_ell_array = create_object_array(__array_size_vec, self.ellipsoid, np.diag(np.array([1, 2, 3, 4, 0])),
                                               np.array([[1]]), 1)
        __is_ans_array = create_object_array(__array_size_vec, np.ones, np.array([1]), np.array([1]), 1)
        __is_test_res = __test_correct(__test_ell_array, __is_ans_array)
        __array_size_vec = np.array([1, 1, 2, 3, 1, 2, 1])
        __diag = np.zeros((100, 100))
        np.fill_diagonal(__diag[50:100, 50:100], 1)
        __test_ell_array = create_object_array(__array_size_vec, self.ellipsoid, __diag, np.array([1]), 1)
        __is_ans_array = create_object_array(__array_size_vec, np.ones, np.array([1]), np.array([1]), 1)
        __is_test_res = __test_correct(__test_ell_array, __is_ans_array)
        assert type(__is_ans_array) == type(__is_test_res)
        __test_ell_array, _, __is_ans_array = create_typical_array(TestEllTCMultiDim(), 16)
        __is_test_res = __test_correct(__test_ell_array, __is_ans_array)
        assert type(__is_ans_array) == type(__is_test_res)
        # Empty self.ellipsoid
        __test_error(1)
        __test_error(14)
        __test_error(15)

    @classmethod
    def __create_typical_array(cls, *args):
        return zip(*[create_typical_array(TestEllTCMultiDim(), x) for x in args])

    def test_is_empty(self):
        def __test_correct(test_ell_array, is_ans_array):
            is_test_res = AEllipsoid.is_empty(test_ell_array).flatten()
            assert np.array_equal(is_test_res, is_ans_array)
            return is_test_res

        __array_size_vec = np.array([2, 1, 1, 1, 1, 3, 1, 1])
        __test_ell_array = create_object_array(__array_size_vec, self.ellipsoid, 1, 1, 0).flatten()
        __is_ans_array = create_object_array(__array_size_vec, np.ones, np.array([1]), np.array([1]), 1).flatten()
        __test_correct(__test_ell_array, __is_ans_array)

        # Check not empty self.ellipsoid
        __test_ell_array, __is_ans_array = create_typical_array(self, 5)
        __test_correct(__test_ell_array, __is_ans_array)

        __array_size_vec = np.array([1, 1, 1, 1, 1, 4, 1, 1, 3])
        __diag = np.zeros((100, 100))
        np.fill_diagonal(__diag[50:100, 50:100], 1)
        __test_ell_array = create_object_array(__array_size_vec, self.ellipsoid, __diag, np.array([1]), 1)
        __is_ans_array = create_object_array(__array_size_vec, np.zeros, np.array([1]), np.array([1]), 1).flatten()
        __test_correct(__test_ell_array, __is_ans_array)

        __test_ell_array, *_, __is_ans_array = create_typical_array(self, 16)
        __is_test_res = __test_correct(__test_ell_array, __is_ans_array)
        assert type(__is_ans_array) == type(__is_test_res)


def create_typical_array(ell_factory_obj: TestEllTCMultiDim, flag):
    array_size_vec = np.array([2, 1, 1, 2, 1, 3, 1])
    if flag == 1:
        array_size_vec = np.array([2, 1, 3, 2, 1, 1, 4])
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid, 1, 1, 0)
        ans_num_array = create_object_array(array_size_vec, np.diag, np.array([0]), np.array([1]), 1).flatten()
        is_ans_array = np.ones(array_size_vec).flatten()
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array, ans_num_array, is_ans_array, error_str
    if flag == 2:
        array_size_vec = np.array([1, 2, 4, 3, 2, 1])
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 1, 1, 1).flatten()
        ans_num_array = create_object_array(array_size_vec, np.diag, np.array([1]), np.array([1]), 1).flatten()
        ans_volume_double_array = create_object_array(array_size_vec, np.diag, np.array([2]),
                                                      np.array([1]), 1).flatten()
        is_ans_array = np.ones(array_size_vec).flatten()
        return test_ell_array, ans_num_array, ans_volume_double_array, is_ans_array
    if flag == 3:
        array_size_vec = np.array([1, 1, 1, 1, 1, 7, 1, 1, 7])
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid,
                                             np.eye(5, 5), np.array([1]), 1).flatten()
        ans_num_array = create_object_array(array_size_vec, np.diag, np.array([5]), np.array([1]), 1).flatten()
        volume_double = 8 * (math.pi**2)/15
        ans_volume_double_array = create_object_array(array_size_vec, np.diag,
                                                      np.array([volume_double]), np.array([1]), 1).flatten()
        return test_ell_array, ans_num_array, ans_volume_double_array
    if flag == 4:
        array_size_vec = np.array([2, 1, 3, 2, 1, 1, 4, 1, 1])
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid,
                                             np.diag(np.array([1, 2, 3, 4, 0])), np.array([1]), 1).flatten()
        ans_dim_num_array = create_object_array(array_size_vec, np.diag, np.array([5]), np.array([1]), 1).flatten()
        ans_rank_num_array = create_object_array(array_size_vec, np.diag, np.array([4]), np.array([1]), 1).flatten()
        ans_volume_double_array = create_object_array(array_size_vec, np.diag,
                                                      np.array([0]), np.array([1]), 1).flatten()
        return test_ell_array, ans_dim_num_array, ans_rank_num_array, ans_volume_double_array
    if flag == 5:
        array_size_vec = np.array([1, 2, 4, 3, 2])
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 1, 1, 1).flatten()
        is_ans_array = create_object_array(array_size_vec, np.zeros, np.array([1]), np.array([1]), 1).flatten()
        return test_ell_array, is_ans_array
    if flag == 6:
        array_size_vec = np.array([1, 1, 2, 3, 2, 1, 1, 1, 4])
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid,
                                             np.zeros((100, 100)), np.eye(1), 1).flatten()
        ans_num_array = create_object_array(array_size_vec, np.diag, np.array([0]), np.eye(1), 1).flatten()
        return test_ell_array, ans_num_array
    if flag == 7:
        array_size_vec = np.array([2, 3, 2, 1, 1, 1, 4, 1, 1])
        my_mat = np.diag(np.linspace(0.0, 100.0, 101))
        test_ell_array = create_object_array(array_size_vec,
                                             ell_factory_obj.ellipsoid, my_mat, np.array([1]), 1).flatten()
        ans_max_num_array = create_object_array(array_size_vec, np.diag, np.array([100]),
                                                np.array([1]), 1).flatten()
        ans_min_num_array = create_object_array(array_size_vec, np.diag, np.array([0]), np.array([1]), 1).flatten()
        ans_trace_num_array = create_object_array(array_size_vec, np.diag,
                                                  np.array([np.sum(np.linspace(0.0, 100.0, 101))]),
                                                  np.array([1]), 1).flatten()
        return test_ell_array, ans_max_num_array, ans_min_num_array, ans_trace_num_array
    if flag == 8:
        array_size_vec = np.array([1, 1, 1, 1, 1, 7, 1, 1, 7])
        my_mat = np.random.rand(10, 10)
        my_mat = my_mat @ my_mat.T
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid, my_mat, np.array([1]), 1)
        ans_max_num_array = create_object_array(array_size_vec, np.diag,
                                                np.array([np.max(np.linalg.eigvalsh(my_mat))]),
                                                np.array([1]), 1).flatten()
        ans_min_num_array = create_object_array(array_size_vec, np.diag,
                                                np.array([np.min(np.linalg.eigvalsh(my_mat))]),
                                                np.array([1]), 1).flatten()
        ans_trace_num_array = create_object_array(array_size_vec, np.diag, np.array([np.trace(my_mat)]),
                                                  np.array([1]), 1).flatten()
        return test_ell_array, ans_max_num_array, ans_min_num_array, ans_trace_num_array
    if flag == 9:
        __MAX_TOL = Properties.get_rel_tol()
        array_size_vec = np.array([1, 1, 1, 1, 1, 7, 1, 1, 7])
        my_1_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 2, 1, 1).flatten()
        my_2_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid,
                                             np.diag([1 + __MAX_TOL, 1 + __MAX_TOL]), np.array([1]), 1).flatten()
        is_ans_array = np.ones(array_size_vec).flatten()
        return my_1_ell_array, my_2_ell_array, is_ans_array
    if flag == 10:
        __MAX_TOL = Properties.get_rel_tol()
        array_size_vec = np.array([1, 1, 2, 1, 1, 1, 2, 1, 1])
        my_1_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 5, 1, 1).flatten()
        my_2_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid,
                                             np.diag(np.tile(([1 + 100 * __MAX_TOL]), (1, 5)).flatten()),
                                             np.array([1]), 1).flatten()
        is_ans_array = np.zeros(array_size_vec).flatten()
        return my_1_ell_array, my_2_ell_array, is_ans_array
    if flag == 11:
        array_size_vec = np.array([1, 1, 3, 1, 1, 1, 2, 1, 1])
        my_1_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 5, 1, 1).flatten()
        my_2_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 4, 1, 1).flatten()
        is_ans_array = np.zeros(array_size_vec).flatten()
        report_str = 'wrongInput:emptyEllipsoid'
        return my_1_ell_array, my_2_ell_array, is_ans_array, report_str
    if flag == 12:
        __MAX_TOL = Properties.get_rel_tol()
        array_size_vec = np.array([1, 1, 2, 1, 1, 1, 1, 1, 2])
        my_1_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 10, 1, 1).flatten()
        my_2_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball,
                                             (2 * __MAX_TOL) * np.ones((10, 1)), np.eye(10, 10), 1).flatten()
        is_ans_array = np.zeros(array_size_vec).flatten()
        report_str = 'wrongInput:emptyEllipsoid'
        return my_1_ell_array, my_2_ell_array, is_ans_array, report_str
    if flag == 13:
        test_ell_array = np.empty([1, 0, 0, 1, 5], dtype=Ellipsoid)
        test_2_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball,
                                               np.array([3]), np.array([1]), 1)
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array, test_2_ell_array, error_str
    if flag == 14:
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 3, 1, 1)
        test_ell_array[1, 0, 0, 1, 0, 2, 0] = ell_factory_obj.ellipsoid()
        test_2_ell_array = create_object_array(array_size_vec, ell_factory_obj.ell_unitball, 3, 1, 1).flatten()
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array.flatten(), test_2_ell_array, error_str
    if flag == 15:
        test_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid, np.array([[3]]),
                                             np.array([[1]]), 1)
        test_ell_array[1, 0, 0, 1, 0, 2, 0] = Ellipsoid()
        test_2_ell_array = create_object_array(array_size_vec, ell_factory_obj.ellipsoid, np.array([[3]]),
                                               np.array([[1]]), 1).flatten()
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array.flatten(), test_2_ell_array, error_str
    if flag == 16:
        array_size_vec = np.array([1, 0, 0, 1, 5])
        test_ell_array = np.empty(array_size_vec, dtype=Ellipsoid).flatten()
        ans_double_array = np.zeros(array_size_vec).flatten()
        is_ans_array = np.ones(array_size_vec).flatten()
        return test_ell_array, ans_double_array, is_ans_array


def create_object_array(array_size_vec, func, first_arg, second_arg, n_arg) -> np.ndarray:
    n_elems = np.prod(array_size_vec)
    object_c_array = []
    if type(first_arg) == int:
        if n_arg == 0:
            object_c_array = [func() for i_elem in range(np.int64(n_elems))]
        elif n_arg == 1:
            object_c_array = [func(first_arg) for i_elem in range(np.int64(n_elems))]
        elif n_arg == 2:
            object_c_array = [[func(first_arg), func(second_arg)] for i_elem in range(np.int64(n_elems))]
    else:
        m = first_arg.shape
        n = second_arg.shape
        if n_arg == 0:
            object_c_array = [func() for i_elem in range(np.int64(n_elems))]
        elif n_arg == 1:
            first_arg_c_array = np.tile(first_arg.flatten(), (n_elems, 1))
            object_c_array = [func(np.reshape(x, m)) for x in first_arg_c_array]
        elif n_arg == 2:
            first_arg_c_array = np.tile([np.array(first_arg.flatten())], (n_elems, 1))
            second_arg_c_array = np.tile([np.array(second_arg.flatten())], (n_elems, 1))
            object_c_array = [[func(np.reshape(x, m)), func(np.reshape(y, n))] for x, y in [list(first_arg_c_array),
                                                                                            list(second_arg_c_array)]]
    return np.reshape(np.array([object_c_array]).flatten(), array_size_vec)


def check_maxeig_ans_mineig(is_maxeig_check):
    def __test_correct(flag):
        if is_maxeig_check:
            test_ell_array, ans_num_array, *_ = create_typical_array(TestEllTCMultiDim(), flag)
            test_num_array = AEllipsoid.max_eig(test_ell_array)
        else:
            if (flag == 2) or (flag == 6) or (flag == 16):
                test_ell_array, ans_num_array, *_ = create_typical_array(TestEllTCMultiDim(), flag)
            else:
                test_ell_array, _, ans_num_array, *_ = create_typical_array(TestEllTCMultiDim(), flag)
            test_num_array = AEllipsoid.min_eig(test_ell_array)
        assert np.array_equal(ans_num_array, test_num_array)
        return ans_num_array, test_num_array

    def __test_error(flag):
        test_ell_array, _, error_str, *_ = create_typical_array(TestEllTCMultiDim(), flag)
        if is_maxeig_check:
            if flag == 1:
                with pytest.raises(Exception) as e:
                    [AEllipsoid.max_eig(x) for x in test_ell_array]
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    AEllipsoid.max_eig(test_ell_array)  # TestEllTCMultiDim.ellipsoid())
                assert error_str in str(e.value)
        else:
            if flag == 1:
                with pytest.raises(Exception) as e:
                    AEllipsoid.min_eig(test_ell_array)
                assert 'wrongInput:emptyEllipsoid' in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    AEllipsoid.min_eig(test_ell_array)
                assert error_str in str(e.value)

    # Check degenerate matrix

    __test_correct(6)
    __test_correct(2)
    __test_correct(7)
    __test_correct(8)
    __ans_num_array, _test_num_array = __test_correct(16)
    assert type(_test_num_array) == type(__ans_num_array)

    __test_error(14)
    __test_error(1)
    __test_error(15)
