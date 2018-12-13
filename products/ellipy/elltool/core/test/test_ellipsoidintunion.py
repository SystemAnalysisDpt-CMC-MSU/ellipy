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
        mode = 'u'
        n_dim = 100
        test_ell_vec = self.ellipsoid(np.zeros((n_dim, 1)), np.eye(n_dim))
        test_point_vec = np.zeros((n_dim, 1))
        test_res_vec = Ellipsoid.is_internal(np.array([[test_ell_vec]]), test_point_vec, mode)
        assert test_res_vec

        test_point_vec[n_dim - 1] = 1
        test_res_vec = Ellipsoid.is_internal(np.array([[test_ell_vec]]), test_point_vec, mode)
        assert test_res_vec

        for i_dim in range(0, n_dim):
            test_point_vec[i_dim] = 1 / np.sqrt(n_dim)

        test_res_vec = Ellipsoid.is_internal(np.array([[test_ell_vec]]), test_point_vec, mode)
        assert test_res_vec

        test_point_vec = np.ones((n_dim, 1))
        test_res_vec = Ellipsoid.is_internal(np.array([[test_ell_vec]]), test_point_vec, mode)
        assert not test_res_vec

        for i_dim in range(0, n_dim):
            test_point_vec[i_dim] = 1 / np.sqrt(n_dim)

        test_point_vec[0] = test_point_vec[0] + 1e-4
        test_res_vec = Ellipsoid.is_internal(np.array([[test_ell_vec]]), test_point_vec, mode)
        assert not test_res_vec

        n_dim = 3
        test_ell_vec = self.ellipsoid(np.zeros((n_dim, 1)),
                                      np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.float64))
        test_point_vec = np.array([[0.3, -0.8, 0]]).T
        test_ell_vec = np.array([[test_ell_vec]])
        test_res_vec = Ellipsoid.is_internal(test_ell_vec, test_point_vec, mode)
        assert test_res_vec

        test_point_vec = np.array([[0.3, -0.8, 1e-3]]).T
        test_res_vec = Ellipsoid.is_internal(test_ell_vec, test_point_vec, mode)
        assert not test_res_vec

        n_dim = 2
        test_ell_vec = np.empty((2, 1), dtype=Ellipsoid)
        test_ell_vec[0] = self.ellipsoid(np.zeros((n_dim, 1)), np.eye(n_dim))
        test_ell_vec[1] = self.ellipsoid(np.array([[2, 0]]).T, np.eye(n_dim))
        test_point_vec = np.array([[1.0, 0], [2, 0]]).T
        test_res_vec = Ellipsoid.is_internal(test_ell_vec, test_point_vec, 'u')
        assert test_res_vec[0] and test_res_vec[1]

        test_res_vec = Ellipsoid.is_internal(test_ell_vec, test_point_vec, 'i')
        assert test_res_vec[0] and not test_res_vec[1]

        test_ell_vec = np.empty((1000, 1), dtype=Ellipsoid)
        for i_num in range(0, 1000):
            test_ell_vec[i_num] = self.ellipsoid(np.eye(2))
        test_point_vec = np.array([[0.0, 0]]).T
        test_res_vec = Ellipsoid.is_internal(test_ell_vec, test_point_vec, 'i')
        assert test_res_vec
        test_res_vec = Ellipsoid.is_internal(test_ell_vec, test_point_vec, 'u')
        assert test_res_vec

    def test_polar(self):
        n_dim = 100
        test_ell_vec = self.ellipsoid(np.zeros((n_dim, 1)), np.eye(n_dim))
        polar_ellipsoid = Ellipsoid.polar(np.array([test_ell_vec]))
        assert Ellipsoid.eq(np.array([test_ell_vec]), polar_ellipsoid.flatten())

        n_dim = 100
        test_sing_ell_vec = self.ellipsoid(np.zeros((n_dim, 1)), np.zeros((n_dim, n_dim)))
        with pytest.raises(Exception) as e:
            Ellipsoid.polar(np.array([test_sing_ell_vec]))
        assert 'degenerateEllipsoid' in str(e.value)

        n_dim = 3
        test_sing_ell_vec = self.ellipsoid(np.zeros((n_dim, 1)),
                                           np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.float64))
        with pytest.raises(Exception) as e:
            Ellipsoid.polar(np.array([test_sing_ell_vec]))
        assert 'degenerateEllipsoid' in str(e.value)

        n_dim = 2
        test_ell_vec = self.ellipsoid(np.zeros((n_dim, 1)), np.array([[2, 0], [0, 1]], dtype=np.float64))
        polar_ell_vec = Ellipsoid.polar(np.array([test_ell_vec]))
        ans_ell_vec = self.ellipsoid(np.zeros((n_dim, 1)), np.array([[0.5, 0], [0, 1]], dtype=np.float64))
        assert Ellipsoid.eq(np.array([ans_ell_vec]), polar_ell_vec.flatten())

        test_ell_vec = self.ellipsoid(np.array([[0], [0.5]], dtype=np.float64), np.eye(2))
        polar_ell_vec = Ellipsoid.polar(np.array([test_ell_vec]))
        ans_ell_vec = self.ellipsoid(np.array([[0], [-2/3]], dtype=np.float64),
                                     np.array([[4/3, 0], [0, 16/9]], dtype=np.float64))
        assert Ellipsoid.eq(np.array([ans_ell_vec]), polar_ell_vec.flatten())
