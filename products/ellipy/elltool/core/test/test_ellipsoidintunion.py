from ellipy.elltool.core.core import *
from ellipy.elltool.core.hyperplane.Hyperplane import *
from ellipy.elltool.core.ellipsoid.Ellipsoid import *
import numpy as np
import pytest


class TestEllipsoidIntUnionTC:
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
        __MODE = 'u'
        __N_DIM = 100
        test_ell_vec = self.ellipsoid(np.zeros((__N_DIM, 1)), np.eye(__N_DIM))
        test_point_vec = np.zeros((__N_DIM, 1))
        test_res_vec = test_ell_vec.is_internal(np.array([[test_ell_vec]]), test_point_vec, __MODE)
        assert test_res_vec

        test_point_vec[__N_DIM - 1] = 1
        test_res_vec = test_ell_vec.is_internal(np.array([[test_ell_vec]]), test_point_vec, __MODE)
        assert test_res_vec

        for i_dim in range(0, __N_DIM):
            test_point_vec[i_dim] = 1 / np.sqrt(__N_DIM)

        test_res_vec = test_ell_vec.is_internal(np.array([[test_ell_vec]]), test_point_vec, __MODE)
        assert test_res_vec

        test_point_vec = np.ones((__N_DIM, 1))
        test_res_vec = test_ell_vec.is_internal(np.array([[test_ell_vec]]), test_point_vec, __MODE)
        assert not test_res_vec

        for i_dim in range(0, __N_DIM):
            test_point_vec[i_dim] = 1 / np.sqrt(__N_DIM)

        test_point_vec[0] = test_point_vec[0] + 1e-4
        test_res_vec = test_ell_vec.is_internal(np.array([[test_ell_vec]]), test_point_vec, __MODE)
        assert not test_res_vec

        __N_DIM = 3
        test_ell_vec = self.ellipsoid(np.zeros((__N_DIM, 1)),
                                      np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.float64))
        test_point_vec = np.array([[0.3, -0.8, 0]]).T
        test_ell_vec = np.array([[test_ell_vec]])
        test_res_vec = test_ell_vec.flat[0].is_internal(test_ell_vec, test_point_vec, __MODE)
        assert test_res_vec

        test_point_vec = np.array([[0.3, -0.8, 1e-3]]).T
        test_res_vec = test_ell_vec.flat[0].is_internal(test_ell_vec, test_point_vec, __MODE)
        assert not test_res_vec

        __N_DIM = 2
        test_ell_vec = np.empty((2, 1), dtype=object)
        test_ell_vec[0] = self.ellipsoid(np.zeros((__N_DIM, 1)), np.eye(__N_DIM))
        test_ell_vec[1] = self.ellipsoid(np.array([[2, 0]]).T, np.eye(__N_DIM))
        test_point_vec = np.array([[1.0, 0], [2, 0]]).T
        test_res_vec = test_ell_vec.flat[0].is_internal(test_ell_vec, test_point_vec, 'u')
        assert test_res_vec[0] and test_res_vec[1]

        test_res_vec = test_ell_vec.flat[0].is_internal(test_ell_vec, test_point_vec, 'i')
        assert test_res_vec[0] and not test_res_vec[1]

        test_ell_vec = np.empty((1000, 1), dtype=object)
        for i_num in range(0, 1000):
            test_ell_vec[i_num] = self.ellipsoid(np.eye(2))
        test_point_vec = np.array([[0.0, 0]]).T
        test_res_vec = test_ell_vec.flat[0].is_internal(test_ell_vec, test_point_vec, 'i')
        assert test_res_vec
        test_res_vec = test_ell_vec.flat[0].is_internal(test_ell_vec, test_point_vec, 'u')
        assert test_res_vec

    def test_polar(self):
        __N_DIM = 100
        test_ell_vec = self.ellipsoid(np.zeros((__N_DIM, 1)), np.eye(__N_DIM))
        polar_ellipsoid = test_ell_vec.polar(np.array([test_ell_vec]))
        assert test_ell_vec.eq(np.array([test_ell_vec]), polar_ellipsoid.flatten())

        __N_DIM = 100
        test_sing_ell_vec = self.ellipsoid(np.zeros((__N_DIM, 1)), np.zeros((__N_DIM, __N_DIM)))
        with pytest.raises(Exception) as e:
            test_sing_ell_vec.polar(np.array([test_sing_ell_vec]))
        assert 'degenerateEllipsoid' in str(e.value)

        __N_DIM = 3
        test_sing_ell_vec = self.ellipsoid(np.zeros((__N_DIM, 1)),
                                           np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.float64))
        with pytest.raises(Exception) as e:
            test_sing_ell_vec.polar(np.array([test_sing_ell_vec]))
        assert 'degenerateEllipsoid' in str(e.value)

        __N_DIM = 2
        test_ell_vec = self.ellipsoid(np.zeros((__N_DIM, 1)), np.array([[2, 0], [0, 1]], dtype=np.float64))
        polar_ell_vec = test_ell_vec.polar(np.array([test_ell_vec]))
        ans_ell_vec = self.ellipsoid(np.zeros((__N_DIM, 1)), np.array([[0.5, 0], [0, 1]], dtype=np.float64))
        assert ans_ell_vec.eq(np.array([ans_ell_vec]), polar_ell_vec.flatten())

        test_ell_vec = self.ellipsoid(np.array([[0], [0.5]], dtype=np.float64), np.eye(2))
        polar_ell_vec = test_ell_vec.polar(np.array([test_ell_vec]))
        ans_ell_vec = self.ellipsoid(np.array([[0], [-2/3]], dtype=np.float64),
                                     np.array([[4/3, 0], [0, 16/9]], dtype=np.float64))
        assert ans_ell_vec.eq(np.array([ans_ell_vec]), polar_ell_vec.flatten())
