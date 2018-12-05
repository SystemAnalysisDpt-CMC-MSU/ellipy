from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.elltool.core.hyperplane.Hyperplane import *
from ellipy.elltool.core.aellipsoid.AEllipsoid import *
import pytest
import numpy as np


class TestEllipsoidTestCase:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def hyperplane(self, *args, **kwargs):
        return Hyperplane(*args, **kwargs)

    def test_max_eig(self):
        __EPS = 1e-16
        with pytest.raises(Exception) as e:
            AEllipsoid.max_eig(np.array([self.ellipsoid()]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Check degenerate matrix
        test_ellipsoid_1 = np.array([self.ellipsoid(np.array([[1], [1]]), np.zeros((2, 2)))])
        test_ellipsoid_2 = np.array([self.ellipsoid(np.zeros((2, 2)))])
        is_test_res = (AEllipsoid.max_eig(test_ellipsoid_1) == 0) and (AEllipsoid.max_eig(test_ellipsoid_2) == 0)
        assert np.array_equal(is_test_res, np.array([True]))
        # Check on diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.diag(np.linspace(1.0, 5.2, 22)))])
        is_test_res = AEllipsoid.max_eig(test_ellipsoid) == 5.2
        assert np.array_equal(is_test_res, np.array([True]))
        # Check on not diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.array([[1, 1, -1], [1, 4, -3], [-1, -3, 9]]))])
        is_test_res = AEllipsoid.max_eig(test_ellipsoid) - max(np.linalg.eigvalsh(np.array([[1, 1, -1], [1, 4, -3],
                                                                                            [-1, -3, 9]]))) <= __EPS
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 13))),
                                 self.ellipsoid(np.array(np.linspace(0.0, 1.4, 15)).T,
                                                np.diag(np.linspace(0.1, 1.5, 15))),
                                 self.ellipsoid(np.random.rand(20, 1), np.diag(np.arange(1, 21)))])
        test_max_eig_vec = AEllipsoid.max_eig(test_ell_vec)
        is_test_res = np.all(test_max_eig_vec == [12, 1.5, 20])
        assert is_test_res

        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.0, 2.0, 21).T, np.diag(np.linspace(0.0, 0.2, 21))),
                                  self.ellipsoid(-10 * np.ones((41, 1)), np.diag(np.linspace(20.0, 420.0, 41))),
                                  self.ellipsoid(np.random.rand(50, 1), 9 * np.eye(50, 50))],
                                 [self.ellipsoid(5 * np.eye(10, 10)),
                                  self.ellipsoid(np.diag(np.linspace(0.0, 0.01, 101))),
                                  self.ellipsoid(np.zeros((30, 30)))]])
        test_max_eig_mat = AEllipsoid.max_eig(test_ell_mat)
        is_test_mat = test_max_eig_mat == [0.2, 420.0, 9.0, 5.0, 0.01, 0.0]
        is_test_res = np.all(is_test_mat)
        assert is_test_res
