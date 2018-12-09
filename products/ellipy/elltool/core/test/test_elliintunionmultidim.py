from ellipy.elltool.core.core import *
from ellipy.elltool.core.hyperplane.Hyperplane import *
import numpy as np
from numpy import matlib as ml
import pytest


class TestElliIntUnionTCMultiDim:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def ell_unitball(self, *args, **kwargs):
        return ell_unitball(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def hyperplane(self, *args, **kwargs):
        return Hyperplane(*args, **kwargs)

    def test_is_internal(self):
        def test_correct(is_two_check, ans_vec):
            test_res_vec = Ellipsoid.is_internal(test_ell_array, test_point_vec, 'i')
            np.all(ans_vec == test_res_vec)
            if is_two_check:
                test_res_vec = Ellipsoid.is_internal(test_ell_array, test_point_vec, 'u')
                np.all(ans_vec == test_res_vec)

        def test_error(flag):
            test_ell_array_in, _, error_str = create_typical_array(flag)
            with pytest.raises(Exception) as e:
                test_ell_array_in.is_internal(test_point_vec)
            assert error_str in str(e.value)

        array_size_vec = np.array([2, 3, 2, 1, 1, 1, 4])
        test_ell_array = create_object_array(array_size_vec, self.ell_unitball, 3, 1, 1)
        test_point_vec = 0.9 * np.eye(3)
        test_correct(True, np.array([1, 1, 1]))

        array_size_vec = np.array([1, 2, 2, 3, 1, 4])
        test_ell_array = create_object_array(array_size_vec, self.ell_unitball, 2, 1, 1)
        test_point_vec = 1.1 * np.eye(2)
        test_correct(True, np.array([0, 0]))

        array_size_vec = np.array([1, 1, 1, 1, 1, 7, 1, 1, 7])
        test_ell_array = create_object_array(array_size_vec, self.ell_unitball, 5, 1, 1)
        test_point_vec = 0.9 * np.eye(5)
        test_correct(True, np.array([1, 1, 1, 1, 1]))

        array_size_vec = np.array([2, 1, 2, 1, 3, 3])
        test_ell_array = create_object_array(array_size_vec, self.ell_unitball, 4, 1, 1)
        test_mat = 0.9 * np.eye(4)
        test_mat = np.concatenate((test_mat, 1.1 * np.eye(4)), axis=1)
        test_point_vec = test_mat
        test_correct(False, np.array([1, 1, 1, 1, 0, 0, 0, 0]))

        test_ell_array, _ = create_typical_array(8)
        test_mat = np.concatenate((0.9 * np.eye(4), 1.9 * np.eye(4), np.zeros((4, 1))), axis=1)
        test_point_vec = test_mat
        test_correct(False, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]))

        test_res_vec_out = Ellipsoid.is_internal(test_ell_array, test_point_vec, 'u')
        # self.flexAssert([1, 1, 1, 1, 1, 1, 1, 1, 1], test_res_vec_out)
        assert np.all(test_res_vec_out)
        test_error(4)
        test_error(5)
        test_error(6)
        test_error(7)


def create_typical_array(flag, *args, **kwargs):
    array_size_vec = np.array([2, 1, 1, 2, 1, 3, 1])
    if flag == 1:
        array_size_vec = np.array([2, 1, 3, 2, 1, 1, 4])
        test_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 3, 1, 1)
        varargout1 = test_ell_array
        varargout2 = ell_unitball(3, *args, **kwargs)
        return tuple((varargout1, varargout2))
    elif flag == 2:
        array_size_vec = np.array([1, 2, 4, 3, 2])
        test_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 2, 1, 1)
        varargout1 = test_ell_array
        varargout2 = ell_unitball(2, *args, **kwargs)
        return tuple((varargout1, varargout2))
    elif flag == 3:
        array_size_vec = np.array([1, 1, 1, 1, 1, 7, 1, 1, 7])
        test_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 4, 1, 1)
        varargout1 = test_ell_array
        varargout2 = ell_unitball(4, *args, **kwargs)
        return tuple((varargout1, varargout2))
    elif flag == 4:
        ell_obj = Ellipsoid()
        test_ell_array = ell_obj.empty(1, 0, 0, 1, 5)
        test2_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 3, 1, 1)
        error_str = 'wrongInput:emptyArray'
        varargout1 = test_ell_array
        varargout2 = test2_ell_array
        varargout3 = error_str
        varargout4 = array_size_vec
        return tuple((varargout1, varargout2, varargout3, varargout4))
    elif flag == 5:
        test_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 3, 1, 1)
        test_ell_array[2, 1, 1, 2, 1, 3, 1] = Ellipsoid()
        test2_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 3, 1, 1)
        error_str = 'wrongInput:emptyEllipsoid'
        varargout1 = test_ell_array
        varargout2 = test2_ell_array
        varargout3 = error_str
        return tuple((varargout1, varargout2, varargout3))
    elif flag == 6:
        test_ell_array = create_object_array(array_size_vec, Ellipsoid(), 3, 1, 1)
        test2_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 3, 1, 1)
        error_str = 'wrongInput:emptyEllipsoid'
        varargout1 = test_ell_array
        varargout2 = test2_ell_array
        varargout3 = error_str
        return tuple((varargout1, varargout2, varargout3))
    elif flag == 7:
        test_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 3, 1, 1)
        test_ell_array[2, 1, 1, 1, 1, 1, 1] = ell_unitball(7, *args, **kwargs)
        test2_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 3, 1, 1)
        error_str = 'wrongSizes'
        varargout1 = test_ell_array
        varargout2 = test2_ell_array
        varargout3 = error_str
        return tuple((varargout1, varargout2, varargout3))
    elif flag == 8:
        test_mat = np.eye(4)
        array_size_vec = np.array([2, 1, 1, 2, 3, 3])
        test_ell_array = create_object_array(array_size_vec, ell_unitball(*args, **kwargs), 4, 1, 1)
        test_ell_array[1, 1, 1, 1, 1, 1] = Ellipsoid(np.array([0, 0, 0, 1]).T, test_mat)
        test_ell_array[1, 1, 1, 1, 1, 2] = Ellipsoid(np.array([0, 0, 0, -1]).T, test_mat)
        test_ell_array[1, 1, 1, 1, 1, 3] = Ellipsoid(np.array([0, 0, 1, 0]).T, test_mat)
        test_ell_array[1, 1, 1, 1, 2, 1] = Ellipsoid(np.array([0, 0, -1, 0]).T, test_mat)
        test_ell_array[1, 1, 1, 1, 2, 2] = Ellipsoid(np.array([0, 1, 0, 0]).T, test_mat)
        test_ell_array[1, 1, 1, 1, 2, 3] = Ellipsoid(np.array([0, -1, 0, 0]).T, test_mat)
        test_ell_array[1, 1, 1, 1, 3, 1] = Ellipsoid(np.array([1, 0, 0, 0]).T, test_mat)
        test_ell_array[1, 1, 1, 1, 3, 2] = Ellipsoid(np.array([-1, 0, 0, 0]).T, test_mat)
        varargout1 = test_ell_array
        varargout2 = array_size_vec
        return tuple((varargout1, varargout2))


def create_object_array(array_size_vec, func, first_arg, second_arg, n_arg):
    n_elems = np.prod(array_size_vec)
    first_arg_c_array = ml.repmat(first_arg, 1, int(n_elems))
    if n_arg == 1:
        object_c_array = func(first_arg_c_array, 'UniformOutput', False)
    else:
        second_arg_c_array = ml.repmat(second_arg, 1, int(n_elems))
        object_c_array = func(first_arg_c_array, second_arg_c_array, 'UniformOutput', False)
    object_array = np.reshape([object_c_array], array_size_vec)
    return object_array
