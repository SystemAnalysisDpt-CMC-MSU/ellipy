from ellipy.elltool.core.core import *
import pytest


class TestEllSecTCMultiDim:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def ell_unitball(self, *args, **kwargs):
        return ell_unitball(*args, **kwargs)

    def test_minksum_ea(self):
        self.__check_minksum_ea_and_minksum_ia(True)

    def test_minksum_ia(self):
        self.__check_minksum_ea_and_minksum_ia(False)

    def __check_minksum_ea_and_minksum_ia(self, is_ea: bool):

        def test_correct(is_not_high_dim: bool, flag: int):
            if is_not_high_dim:
                my_ell_array, my_mat, ans_ell_vec = self.__create_typical_array(flag)
            else:
                my_ell_array, my_mat, ans_ell_vec = self.__create_typical_high_dim_array(flag)
            if is_ea:
                _compare_for_mink_func('minksum_ea', 2, ans_ell_vec, my_ell_array, my_mat)
            else:
                _compare_for_mink_func('minksum_ia', 2, ans_ell_vec, my_ell_array, my_mat)

        def test_error(flag: int):
            ell_array, _, error_str = _create_typical_array(self, flag)
            if is_ea:
                with pytest.raises(Exception) as e:
                    ell = self.ellipsoid()
                    ell.minksum_ea(ell_array, np.eye(3))
                assert error_str in str(e.value)
            else:
                with pytest.raises(Exception) as e:
                    ell = self.ellipsoid()
                    ell.minksum_ea(ell_array, np.eye(3))
                assert error_str in str(e.value)

        test_correct(True, 4)
        test_correct(True, 5)
        test_correct(True, 6)
        test_correct(False, 2)
        test_error(10)
        test_error(11)
        test_error(12)
        test_error(13)

    def __create_typical_array(self, *args):
        return _create_typical_array(self, *args)

    def __create_typical_high_dim_array(self, *args):
        return _create_typical_high_dim_array(self, *args)


def __create_object_array(array_size: np.ndarray, func, first_arg, second_arg, n_arg: int):
    n_elems = np.prod(array_size).flat[0]
    if n_arg == 0:
        object_list = [func() for _ in range(n_elems)]
    elif n_arg == 1:
        object_list = [func(first_arg) for _ in range(n_elems)]
    elif n_arg == 2:
        object_list = [func(first_arg, second_arg) for _ in range(n_elems)]
    else:
        object_list = np.empty(n_elems)
    return np.reshape(np.array(object_list), array_size)


def _create_typical_high_dim_array(ell_factory_obj, flag: int):
    if flag == 2:
        array_size = np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 1])
        my_ell_array = __create_object_array(array_size, lambda x, y: ell_factory_obj.ellipsoid(x, y),
                                             np.ones(100, dtype=np.float64),
                                             np.array(np.diag(0.25 * np.ones(100)), dtype=np.float64), 2)
        my_mat = np.array(np.vstack((np.eye(5), np.zeros([95, 5]))), dtype=np.float64)
        ans_ell_mat = np.array(np.diag((4.5 ** 2) * np.ones(100)), dtype=np.float64)
        ans_ell_vec = __create_object_array(np.array([5]), lambda x, y: ell_factory_obj.ellipsoid(x, y),
                                            9 * np.ones(100, dtype=np.float64), ans_ell_mat, 2)
        return my_ell_array, my_mat, ans_ell_vec


def _create_typical_array(ell_factory_obj, flag: int):
    if flag == 4:
        array_size = np.array([2, 1, 3, 1, 1, 1, 2])
        my_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 10, 1, 1)
        my_mat = np.array(np.eye(10), dtype=np.float64)
        ans_ell_mat = np.array(np.diag(12 ** 2 * np.ones(10)), dtype=np.float64)
        ans_ell_vec = __create_object_array(np.array([10]), lambda x: ell_factory_obj.ellipsoid(x), ans_ell_mat, 1, 1)
        return my_ell_array, my_mat, ans_ell_vec
    if flag == 5:
        array_size = np.array([1, 2, 1, 3, 2, 1])
        my_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 7, 1, 1)
        my_ell_array[0, 1, 0, 2, 1, 0] = ell_factory_obj.ellipsoid(
            np.array(5 * np.ones(7), dtype=np.float64), np.array(np.diag(9 * np.ones(7)), dtype=np.float64))
        my_mat = np.array(np.hstack((np.eye(7), -np.eye(7))), dtype=np.float64)
        ans_ell_mat = np.array(np.diag(14 ** 2 * np.ones(7)), dtype=np.float64)
        ans_ell_vec = __create_object_array(np.array([14]), lambda x, y: ell_factory_obj.ellipsoid(x, y),
                                            np.array(5 * np.ones(7), dtype=np.float64), ans_ell_mat, 2)
        return my_ell_array, my_mat, ans_ell_vec
    if flag == 6:
        array_size = np.array([1, 1, 1, 1, 1, 3, 1, 1, 2])
        my_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 1, 1, 1)
        my_ell_array[0, 0, 0, 0, 0, 1, 0, 0, 0] = ell_factory_obj.ellipsoid(
            np.array(np.array([-1]), dtype=np.float64), np.array([[0.25]], dtype=np.float64))
        my_mat = np.array([[1, -1]], dtype=np.float64)
        ans_ell_mat = np.array([[5.5 ** 2]], dtype=np.float64)
        ans_ell_vec = __create_object_array(np.array([2]), lambda x, y: ell_factory_obj.ellipsoid(x, y),
                                            np.array([-1], dtype=np.float64),
                                            ans_ell_mat, 2)
        return my_ell_array, my_mat, ans_ell_vec
    if flag == 10:
        array_size = np.array([2, 1, 1, 2, 1, 3, 1])
        test_ell_array = np.empty([1, 0, 0, 1, 5], dtype=object)
        test2_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 3, 1, 1)
        error_str = 'wrongInput:emptyArray'
        return test_ell_array, test2_ell_array, error_str
    if flag == 11:
        array_size = np.array([2, 1, 1, 2, 1, 3, 1])
        test_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 3, 1, 1)
        test_ell_array[1, 0, 0, 1, 0, 2, 0] = ell_factory_obj.ellipsoid()
        test2_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 3, 1, 1)
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array, test2_ell_array, error_str
    if flag == 12:
        array_size = np.array([2, 1, 1, 2, 1, 3, 1])
        test_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ellipsoid(), 3, 1, 1)
        test2_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 3, 1, 1)
        error_str = 'wrongInput:emptyEllipsoid'
        return test_ell_array, test2_ell_array, error_str
    if flag == 13:
        array_size = np.array([2, 1, 1, 2, 1, 3, 1])
        test_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 3, 1, 1)
        test2_ell_array = __create_object_array(array_size, lambda x: ell_factory_obj.ell_unitball(x), 3, 1, 1)
        test_ell_array[1, 0, 0, 0, 0, 0, 0] = ell_factory_obj.ell_unitball(7)
        error_str = 'wrongSizes'
        return test_ell_array, test2_ell_array, error_str


def _compare_for_mink_func(method: str, n_arg: int, ans_ell_vec: np.ndarray, first_arg, second_arg,
                           third_arg=0, fourth_arg=0):
    if n_arg == 2:
        res_ell_vec = getattr(first_arg.flat[0], method)(first_arg, second_arg)
    elif n_arg == 3:
        res_ell_vec = getattr(first_arg.flat[0], method)(first_arg, second_arg, third_arg)
    elif n_arg == 4:
        res_ell_vec = getattr(first_arg.flat[0], method)(first_arg, second_arg, third_arg, fourth_arg)
    else:
        res_ell_vec = np.empty(ans_ell_vec.shape)

    is_eq, report_str = res_ell_vec.flat[0].is_equal(res_ell_vec, ans_ell_vec)
    assert all(is_eq) is True, report_str
