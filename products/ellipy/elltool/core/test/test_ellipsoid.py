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
        # Check empty self.ellipsoid
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

    def test_min_eig(self):
        __EPS = 1e-16
        # Check empty self.ellipsoid
        with pytest.raises(Exception) as e:
            AEllipsoid.min_eig(np.array([self.ellipsoid()]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Check degenerate matrix
        test_ellipsoid_1 = np.array([self.ellipsoid(np.array([[-2], [-2]]), np.zeros((2, 2)))])
        test_ellipsoid_2 = np.array([self.ellipsoid(np.zeros((2, 2)))])
        is_test_res = (AEllipsoid.min_eig(test_ellipsoid_1) == 0) and (AEllipsoid.min_eig(test_ellipsoid_2) == 0)
        assert np.array_equal(is_test_res, np.array([True]))
        # Check on diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.diag(np.linspace(4.0, 1.2, 15)))])
        is_test_res = AEllipsoid.min_eig(test_ellipsoid) == 1.2
        assert np.array_equal(is_test_res, np.array([True]))
        # Check on not diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.array([[1, 1, -1], [1, 4, -4], [-1, -4, 9]]))])
        is_test_res = AEllipsoid.min_eig(test_ellipsoid) - np.min(np.linalg.eigvalsh(np.array([[1, 1, -1], [1, 4, -4],
                                                                                               [-1, -4, 9]]))) <= __EPS
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 13))),
                                 self.ellipsoid(np.array(np.linspace(0.0, 1.4, 15)).T,
                                                np.diag(np.linspace(0.1, 1.5, 15))),
                                 self.ellipsoid(np.random.rand(21, 1), np.diag(np.arange(0, 21)))])
        test_min_eig_vec = AEllipsoid.min_eig(test_ell_vec)
        is_test_res = np.all(test_min_eig_vec == [1.0, 0.1, 0.0])
        assert is_test_res

        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.1, 2.0, 20).T, np.diag(np.linspace(0.01, 0.2, 20))),
                                  self.ellipsoid(-10 * np.ones((41, 1)), np.diag(np.linspace(20.0, 420.0, 41))),
                                  self.ellipsoid(np.random.rand(50, 1), 9 * np.eye(50, 50))],
                                 [self.ellipsoid(np.tile(np.diag(np.arange(1, 21)), (2, 2))),
                                  self.ellipsoid(np.diag(np.linspace(0.0001, 0.01, 100))),
                                  self.ellipsoid(np.zeros((30, 30)))]])
        test_min_eig_mat = AEllipsoid.min_eig(test_ell_mat)
        is_test_mat = test_min_eig_mat == [0.01, 20.0, 9.0, 0.0, 0.0001, 0.0]
        is_test_res = np.all(is_test_mat)
        assert is_test_res

    def test_trace(self):
        # Check empty self.ellipsoid
        with pytest.raises(Exception) as e:
            AEllipsoid.trace(np.array([self.ellipsoid()]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Not empty self.ellipsoid
        test_ellipsoid = np.array([self.ellipsoid(np.zeros((10, 1)), np.eye(10, 10))])
        is_test_res = AEllipsoid.trace(test_ellipsoid) == 10.0
        assert np.array_equal(is_test_res, np.array([True]))

        test_ellipsoid = np.array([self.ellipsoid(-np.eye(3, 1), np.array([[1, 0, 1], [0, 0, 0], [1, 0, 2]]))])
        is_test_res = AEllipsoid.trace(test_ellipsoid) == 3.0
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 13))),
                                 self.ellipsoid(np.array(np.linspace(0.0, 1.4, 15)).T,
                                                np.diag(np.linspace(0.1, 1.5, 15))),
                                 self.ellipsoid(np.random.rand(21, 1), np.diag(np.arange(0, 21)))])
        test_trace_vec = AEllipsoid.trace(test_ell_vec)
        is_test_res = np.all(test_trace_vec == [78.0, 12.0, 210.0])
        assert is_test_res

        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.1, 2.0, 20).T, np.diag(np.linspace(0.01, 0.2, 20))),
                                  self.ellipsoid(-10 * np.ones((41, 1)), np.diag(np.linspace(20.0, 420.0, 41))),
                                  self.ellipsoid(np.random.rand(50, 1), 9 * np.eye(50, 50))],
                                 [self.ellipsoid(np.tile(np.diag(np.arange(1, 21)), (2, 2))),
                                  self.ellipsoid(np.diag(np.linspace(0.0001, 0.01, 100))),
                                  self.ellipsoid(np.zeros((30, 30)))]])
        test_trace_mat = AEllipsoid.trace(test_ell_mat)
        is_test_mat = (test_trace_mat == [np.sum(np.linspace(0.01, 0.2, 20)), np.sum(np.linspace(20.0, 420.0, 41)),
                                          9*50, 2 * np.sum(np.arange(1, 21)), np.sum(np.linspace(0.0001, 0.01, 100)),
                                          0.0])
        is_test_res = np.all(is_test_mat)
        assert is_test_res

    def test_is_degenerate(self):
        # Empty self.ellipsoid
        with pytest.raises(Exception) as e:
            AEllipsoid.is_degenerate(np.array([self.ellipsoid()]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Not degerate self.ellipsoid
        test_ellipsoid = np.array([self.ellipsoid(np.ones((6, 1)), np.eye(6, 6))])
        is_test_res = AEllipsoid.is_degenerate(test_ellipsoid)
        assert np.array_equal(is_test_res, np.array([False]))
        # Degenerate self.ellipsoids
        test_ellipsoid = np.array([self.ellipsoid(np.ones((6, 1)), np.zeros((6, 6)))])
        is_test_res = AEllipsoid.is_degenerate(test_ellipsoid)
        assert np.array_equal(is_test_res, np.array([True]))

        test_a_mat = np.array([[3, 1], [0, 1], [-2, 1]])
        test_ellipsoid = np.array([self.ellipsoid(test_a_mat@test_a_mat.T)])
        is_test_res = AEllipsoid.is_degenerate(test_ellipsoid)
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 23))),
                                 self.ellipsoid(np.linspace(0.0, 1.4, 15).T, np.diag(np.arange(1, 16))),
                                 self.ellipsoid(np.random.rand(21, 1), np.diag(np.arange(0, 21)))])
        is_test_deg_vec = AEllipsoid.is_degenerate(test_ell_vec)
        is_test_res = np.all(is_test_deg_vec == [False, False, True])
        assert is_test_res
        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.0, 2.0, 21).T, np.diag(np.linspace(0.0, 0.2, 21))),
                                  self.ellipsoid(np.eye(40, 40)),
                                  self.ellipsoid(np.random.rand(50, 1), 9*np.eye(50, 50))],
                                 [self.ellipsoid(np.diag(np.linspace(10.0, 40.0, 16))),
                                  self.ellipsoid(np.tile(np.concatenate((np.diag(np.linspace(0.0, 5.0, 51)),
                                                                         np.diag(np.linspace(0.0, 5.0, 51))),
                                                                        axis=1), (2, 1))),
                                  self.ellipsoid(np.zeros((30, 30)))]])
        is_test_deg_mat = AEllipsoid.is_degenerate(test_ell_mat)
        is_test_mat = (is_test_deg_mat == np.array([[True, False, False], [False, True, True]]))
        is_test_res = np.all(is_test_mat.flatten())
        assert is_test_res

    def test_is_empty(self):
        # Check really empty self.ellipsoid
        test_ellipsoid = self.ellipsoid()
        is_test_res = test_ellipsoid.is_empty(np.array([test_ellipsoid]))
        assert is_test_res
        # Check not empty self.ellipsoid
        test_ellipsoid = self.ellipsoid(np.eye(10, 1), np.eye(10, 10))
        is_test_res = test_ellipsoid.is_empty(np.array([test_ellipsoid]))
        assert not is_test_res
        # High-dimensional self.ellipsoids
        test_ell_vec = [self.ellipsoid(np.diag(np.arange(1, 23))),
                        self.ellipsoid(np.linspace(0.0, 1.4, 15).T, np.diag(np.arange(1, 16))),
                        self.ellipsoid(np.random.rand(21, 1), np.diag(np.arange(0, 21))),
                        self.ellipsoid(), self.ellipsoid(), self.ellipsoid(np.zeros((40, 40)))]
        is_test_emp_vec = np.array([ell_obj.is_empty(np.array([ell_obj])) for ell_obj in list(test_ell_vec)]).flatten()
        is_test_res = np.all(is_test_emp_vec == [False, False, False, True, True, False])
        assert is_test_res

        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.0, 2.0, 21).T, np.diag(np.linspace(0.0, 0.2, 21))),
                                  self.ellipsoid(np.eye(40, 40)),
                                  self.ellipsoid()],
                                 [self.ellipsoid(),
                                  self.ellipsoid(np.tile(np.concatenate((np.diag(np.linspace(0.0, 5.0, 51)),
                                                                         np.diag(np.linspace(0.0, 5.0, 51))),
                                                                        axis=1), (2, 1))),
                                  self.ellipsoid(np.zeros((30, 30)))]]).flatten()
        is_test_emp_mat = np.array([ell_obj.is_empty(np.array(ell_obj)) for ell_obj in list(test_ell_mat)])
        is_test_mat = (is_test_emp_mat == [False, False, True, True, False, False])
        is_test_res = np.all(is_test_mat)
        assert is_test_res
