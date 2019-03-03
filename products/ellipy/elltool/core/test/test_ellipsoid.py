from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.elltool.core.hyperplane.Hyperplane import *
from ellipy.elltool.conf.properties.Properties import *
import pytest
import scipy.io
import numpy as np
import os


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
            ell_obj = self.ellipsoid()
            ell_obj.max_eig(np.array([ell_obj]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Check degenerate matrix
        test_ellipsoid_1 = np.array([self.ellipsoid(np.array([[1], [1]]), np.zeros((2, 2)))])
        test_ellipsoid_2 = np.array([self.ellipsoid(np.zeros((2, 2)))])
        is_test_res = (test_ellipsoid_1.flat[0].max_eig(test_ellipsoid_1) == 0) and \
                      (test_ellipsoid_2.flat[0].max_eig(test_ellipsoid_2) == 0)
        assert np.array_equal(is_test_res, np.array([True]))
        # Check on diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.diag(np.linspace(1.0, 5.2, 22)))])
        is_test_res = test_ellipsoid.flat[0].max_eig(test_ellipsoid) == 5.2
        assert np.array_equal(is_test_res, np.array([True]))
        # Check on not diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.array([[1, 1, -1], [1, 4, -3], [-1, -3, 9]]))])
        is_test_res = test_ellipsoid.flat[0].max_eig(test_ellipsoid) - \
            max(np.linalg.eigvalsh(np.array([[1, 1, -1], [1, 4, -3], [-1, -3, 9]]))) <= __EPS
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 13))),
                                 self.ellipsoid(np.array(np.linspace(0.0, 1.4, 15)).T,
                                                np.diag(np.linspace(0.1, 1.5, 15))),
                                 self.ellipsoid(np.random.rand(20, 1), np.diag(np.arange(1, 21)))])
        test_max_eig_vec = test_ell_vec.flat[0].max_eig(test_ell_vec)
        is_test_res = np.all(test_max_eig_vec == [12, 1.5, 20])
        assert is_test_res

        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.0, 2.0, 21).T, np.diag(np.linspace(0.0, 0.2, 21))),
                                  self.ellipsoid(-10 * np.ones((41, 1)), np.diag(np.linspace(20.0, 420.0, 41))),
                                  self.ellipsoid(np.random.rand(50, 1), 9 * np.eye(50, 50))],
                                 [self.ellipsoid(5 * np.eye(10, 10)),
                                  self.ellipsoid(np.diag(np.linspace(0.0, 0.01, 101))),
                                  self.ellipsoid(np.zeros((30, 30)))]])
        test_max_eig_mat = test_ell_mat.flat[0].max_eig(test_ell_mat)
        is_test_mat = test_max_eig_mat == [0.2, 420.0, 9.0, 5.0, 0.01, 0.0]
        is_test_res = np.all(is_test_mat)
        assert is_test_res

    def test_min_eig(self):
        __EPS = 1e-16
        # Check empty self.ellipsoid
        with pytest.raises(Exception) as e:
            ell_obj = self.ellipsoid()
            ell_obj.min_eig(np.array([ell_obj]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Check degenerate matrix
        test_ellipsoid_1 = np.array([self.ellipsoid(np.array([[-2], [-2]]), np.zeros((2, 2)))])
        test_ellipsoid_2 = np.array([self.ellipsoid(np.zeros((2, 2)))])
        is_test_res = (test_ellipsoid_1.flat[0].min_eig(test_ellipsoid_1) == 0) and \
                      (test_ellipsoid_2.flat[0].min_eig(test_ellipsoid_2) == 0)
        assert np.array_equal(is_test_res, np.array([True]))
        # Check on diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.diag(np.linspace(4.0, 1.2, 15)))])
        is_test_res = test_ellipsoid.flat[0].min_eig(test_ellipsoid) == 1.2
        assert np.array_equal(is_test_res, np.array([True]))
        # Check on not diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.array([[1, 1, -1], [1, 4, -4], [-1, -4, 9]]))])
        is_test_res = test_ellipsoid.flat[0].min_eig(test_ellipsoid) - \
            np.min(np.linalg.eigvalsh(np.array([[1, 1, -1], [1, 4, -4], [-1, -4, 9]]))) <= __EPS
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 13))),
                                 self.ellipsoid(np.array(np.linspace(0.0, 1.4, 15)).T,
                                                np.diag(np.linspace(0.1, 1.5, 15))),
                                 self.ellipsoid(np.random.rand(21, 1), np.diag(np.arange(0, 21)))])
        test_min_eig_vec = test_ell_vec.flat[0].min_eig(test_ell_vec)
        is_test_res = np.all(test_min_eig_vec == [1.0, 0.1, 0.0])
        assert is_test_res

        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.1, 2.0, 20).T, np.diag(np.linspace(0.01, 0.2, 20))),
                                  self.ellipsoid(-10 * np.ones((41, 1)), np.diag(np.linspace(20.0, 420.0, 41))),
                                  self.ellipsoid(np.random.rand(50, 1), 9 * np.eye(50, 50))],
                                 [self.ellipsoid(np.tile(np.diag(np.arange(1, 21)), (2, 2))),
                                  self.ellipsoid(np.diag(np.linspace(0.0001, 0.01, 100))),
                                  self.ellipsoid(np.zeros((30, 30)))]])
        test_min_eig_mat = test_ell_mat.flat[0].min_eig(test_ell_mat)
        is_test_mat = test_min_eig_mat == [0.01, 20.0, 9.0, 0.0, 0.0001, 0.0]
        is_test_res = np.all(is_test_mat)
        assert is_test_res

    def test_trace(self):
        # Check empty self.ellipsoid
        with pytest.raises(Exception) as e:
            ell_obj = self.ellipsoid()
            ell_obj.trace(np.array([ell_obj]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Not empty self.ellipsoid
        test_ellipsoid = np.array([self.ellipsoid(np.zeros((10, 1)), np.eye(10, 10))])
        is_test_res = test_ellipsoid.flat[0].trace(test_ellipsoid) == 10.0
        assert np.array_equal(is_test_res, np.array([True]))

        test_ellipsoid = np.array([self.ellipsoid(-np.eye(3, 1), np.array([[1, 0, 1], [0, 0, 0], [1, 0, 2]]))])
        is_test_res = test_ellipsoid.flat[0].trace(test_ellipsoid) == 3.0
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 13))),
                                 self.ellipsoid(np.array(np.linspace(0.0, 1.4, 15)).T,
                                                np.diag(np.linspace(0.1, 1.5, 15))),
                                 self.ellipsoid(np.random.rand(21, 1), np.diag(np.arange(0, 21)))])
        test_trace_vec = test_ell_vec.flat[0].trace(test_ell_vec)
        is_test_res = np.all(test_trace_vec == [78.0, 12.0, 210.0])
        assert is_test_res

        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.1, 2.0, 20).T, np.diag(np.linspace(0.01, 0.2, 20))),
                                  self.ellipsoid(-10 * np.ones((41, 1)), np.diag(np.linspace(20.0, 420.0, 41))),
                                  self.ellipsoid(np.random.rand(50, 1), 9 * np.eye(50, 50))],
                                 [self.ellipsoid(np.tile(np.diag(np.arange(1, 21)), (2, 2))),
                                  self.ellipsoid(np.diag(np.linspace(0.0001, 0.01, 100))),
                                  self.ellipsoid(np.zeros((30, 30)))]])
        test_trace_mat = test_ell_mat.flat[0].trace(test_ell_mat)
        is_test_mat = (test_trace_mat == [np.sum(np.linspace(0.01, 0.2, 20)), np.sum(np.linspace(20.0, 420.0, 41)),
                                          9*50, 2 * np.sum(np.arange(1, 21)), np.sum(np.linspace(0.0001, 0.01, 100)),
                                          0.0])
        is_test_res = np.all(is_test_mat)
        assert is_test_res

    def test_is_degenerate(self):
        # Empty self.ellipsoid
        with pytest.raises(Exception) as e:
            ell_obj = self.ellipsoid()
            ell_obj.is_degenerate(np.array([ell_obj]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Not degerate self.ellipsoid
        test_ellipsoid = np.array([self.ellipsoid(np.ones((6, 1)), np.eye(6, 6))])
        is_test_res = test_ellipsoid.flat[0].is_degenerate(test_ellipsoid)
        assert np.array_equal(is_test_res, np.array([False]))
        # Degenerate self.ellipsoids
        test_ellipsoid = np.array([self.ellipsoid(np.ones((6, 1)), np.zeros((6, 6)))])
        is_test_res = test_ellipsoid.flat[0].is_degenerate(test_ellipsoid)
        assert np.array_equal(is_test_res, np.array([True]))

        test_a_mat = np.array([[3, 1], [0, 1], [-2, 1]])
        test_ellipsoid = np.array([self.ellipsoid(test_a_mat @ test_a_mat.T)])
        is_test_res = test_ellipsoid.flat[0].is_degenerate(test_ellipsoid)
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 23))),
                                 self.ellipsoid(np.linspace(0.0, 1.4, 15).T, np.diag(np.arange(1, 16))),
                                 self.ellipsoid(np.random.rand(21, 1), np.diag(np.arange(0, 21)))])
        is_test_deg_vec = test_ell_vec.flat[0].is_degenerate(test_ell_vec)
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
        is_test_deg_mat = test_ell_mat.flat[0].is_degenerate(test_ell_mat)
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
        test_ell_vec = np.array([self.ellipsoid(np.diag(np.arange(1, 23))),
                                 self.ellipsoid(np.linspace(0.0, 1.4, 15).T, np.diag(np.arange(1, 16))),
                                 self.ellipsoid(np.random.rand(21, 1), np.diag(np.arange(0, 21))),
                                 self.ellipsoid(), self.ellipsoid(), self.ellipsoid(np.zeros((40, 40)))])
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

    def test_volume(self):
        __ABS_TOL = Properties.get_abs_tol()
        # Check empty self.ellipsoid
        with pytest.raises(Exception) as e:
            ell_obj = self.ellipsoid()
            ell_obj.volume(np.array([ell_obj]))
        assert 'wrongInput:emptyEllipsoid' in str(e.value)
        # Check degenerate self.ellipsoid
        test_ellipsoid = np.array([self.ellipsoid(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))])
        is_test_res = test_ellipsoid.flat[0].volume(test_ellipsoid) == 0
        assert np.array_equal(is_test_res.flatten(), np.array([True]))
        # Check dim=1 with two different centers
        test_ellipsoid_1 = np.array([self.ellipsoid(np.array([[2]]), np.array([[1]]))])
        test_ellipsoid_2 = np.array([self.ellipsoid(np.array([[1]]))])
        is_test_res = (test_ellipsoid_1.flat[0].volume(test_ellipsoid_1) == 2) and \
                      (test_ellipsoid_2.flat[0].volume(test_ellipsoid_2) == 2)
        assert np.array_equal(is_test_res, np.array([True]))
        # Check dim=2 with two different centers
        test_ellipsoid_1 = np.array([self.ellipsoid(np.array([[1], [-1]]), np.eye(2, 2))])
        test_ellipsoid_2 = np.array([self.ellipsoid(np.eye(2, 2))])
        is_test_res = ((test_ellipsoid_1.flat[0].volume(test_ellipsoid_1) - np.pi) <= __ABS_TOL) and \
                      ((test_ellipsoid_2.flat[0].volume(test_ellipsoid_2) - np.pi) <= __ABS_TOL)
        assert np.array_equal(is_test_res, np.array([True]))
        # Check dim=3 with not diagonal matrix
        test_ellipsoid = np.array([self.ellipsoid(np.array([[1, 1, -1], [1, 4, -3], [-1, -3, 9]]))])
        is_test_res = ((test_ellipsoid.flat[0].volume(test_ellipsoid) - (8 * np.sqrt(5) * np.pi / 3)) <= __ABS_TOL)
        assert np.array_equal(is_test_res, np.array([True]))
        # Check dim=5
        test_ellipsoid = np.array([self.ellipsoid(4 * np.ones((5, 1)), np.eye(5, 5))])
        is_test_res = ((test_ellipsoid.flat[0].volume(test_ellipsoid) - (8 * np.pi ** 2 / 15)) <= __ABS_TOL)
        assert np.array_equal(is_test_res, np.array([True]))
        # Check dim=6
        test_ellipsoid = np.array([self.ellipsoid(-np.ones((6, 1)), np.diag(np.array([1, 4, 9, 16, 1, 25])))])
        is_test_res = ((test_ellipsoid.flat[0].volume(test_ellipsoid) - (20 * np.pi ** 3)) <= __ABS_TOL)
        assert np.array_equal(is_test_res, np.array([True]))
        # High-dimensional self.ellipsoids
        test_ell_mat = np.array([[self.ellipsoid(np.linspace(0.1, 2, 20).T, np.diag(np.linspace(0.01, 0.2, 20))),
                                  self.ellipsoid(-10*np.ones((13, 1)), np.diag(np.linspace(0.1, 1.3, 13)))],
                                 [self.ellipsoid(np.random.rand(20, 1), 9 * np.diag(np.arange(0, 20))),
                                  self.ellipsoid(np.diag(np.arange(1, 22)))],
                                 [self.ellipsoid(np.diag(np.linspace(0.1, 10, 100))),
                                  self.ellipsoid(np.diag(np.linspace(0.0, 0.01, 101)))]])
        test_vol_mat = test_ell_mat.flat[0].volume(test_ell_mat)
        test_right_vol_mat = np.array([[(np.pi**6) * np.sqrt(np.prod(np.linspace(0.01, 0.2, 20)) /
                                                             np.prod(np.arange(1, 7))),
                                        (np.pi**6) * (2**7) * np.sqrt(np.prod(np.linspace(0.1, 1.3, 13))) /
                                        np.prod(np.linspace(1, 13, 7))],
                                       [0.0,
                                        (np.pi ** 10) * (2 ** 11) * np.sqrt(np.prod(np.linspace(1.0, 21.0, 21))) /
                                        np.prod(np.linspace(1, 21, 11))],
                                       [(np.pi ** 50) * np.sqrt(np.prod(np.linspace(0.1, 10, 10))) /
                                        np.prod(np.linspace(1.0, 50.0, 50)),
                                        0.0]])
        is_test_eq_mat = (test_vol_mat - test_right_vol_mat.flatten()) <= __ABS_TOL
        is_test_res = np.all(is_test_eq_mat.flatten())
        assert is_test_res

    def test_get_grid_by_factor(self):
        __GET_GRID_BY_FACTOR = \
            scipy.io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'get_grid_by_factor_data.mat'))['getGridByFactorData']

        class MyTestClass(Ellipsoid):
            def call_method(self, *args):
                return self._get_grid_by_factor(*args)

        ell_obj = MyTestClass(np.diag([1, 2, 1]))
        v_grid_mat, f_grid_mat = ell_obj.call_method()
        assert np.allclose(__GET_GRID_BY_FACTOR[0][0][0][0], v_grid_mat, rtol=1e-9)
        assert np.allclose(__GET_GRID_BY_FACTOR[0][0][0][1], f_grid_mat + 1, rtol=1e-9)

        ell_obj = MyTestClass(np.diag([0.8, 0.1, 0.1]))
        v_grid_mat, f_grid_mat = ell_obj.call_method(np.array(1., dtype=np.float64))
        assert np.allclose(__GET_GRID_BY_FACTOR[0][1][0][0], v_grid_mat, rtol=1e-9)
        assert np.allclose(__GET_GRID_BY_FACTOR[0][1][0][1], f_grid_mat + 1, rtol=1e-9)

        ell_obj = MyTestClass(np.diag([1, 2, 1]))
        v_grid_mat, f_grid_mat = ell_obj.call_method(np.array([2., 4.], dtype=np.float64))
        assert np.allclose(__GET_GRID_BY_FACTOR[0][2][0][0], v_grid_mat, rtol=1e-9)
        assert np.allclose(__GET_GRID_BY_FACTOR[0][2][0][1], f_grid_mat + 1, rtol=1e-9)

        ell_obj = MyTestClass(np.diag([0.8, 0.1, 0.1]))
        v_grid_mat, f_grid_mat = ell_obj.call_method(np.array([10., 5.], dtype=np.float64))
        assert np.allclose(__GET_GRID_BY_FACTOR[0][3][0][0], v_grid_mat, rtol=1e-9)
        assert np.allclose(__GET_GRID_BY_FACTOR[0][3][0][1], f_grid_mat + 1, rtol=1e-9)

    def test_sqrtm_pos_tolerance_failure(self):
        sh1_mat = np.diag([1e-7, 1e-7, 1e-7, 1e-7]) + np.diag([1, 1, 0, 0])
        sh2_mat = np.eye(4, dtype=np.float64)
        ell_arr = np.array([self.ellipsoid(np.zeros(4), sh1_mat), self.ellipsoid(np.zeros(4), sh2_mat)])
        ell_arr[0].minksum_ia(ell_arr, np.array([0, 0, 1, 0]))

    def test_get_copy(self):
        def is_equal(a1, a2):
            if np.shape(a1) != np.shape(a2):
                return 0
            else:
                eps = 1e-9
                is_eq_mat = np.zeros(np.shape(a1))
                for i in range(np.shape(a1)[0]):
                    for j in range(np.shape(a1)[1]):
                        if np.shape(a1[i][j].get_shape_mat()) == np.shape(a2[i][j].get_shape_mat()):
                            is_eq_mat[i][j] = (np.all(np.abs((a1[i][j]).get_center_vec() -
                                                             (a2[i][j]).get_center_vec()) < eps) and
                                               np.all(np.abs((a1[i][j]).get_shape_mat() -
                                                             (a2[i][j]).get_shape_mat()) < eps))
                        else:
                            is_eq_mat[i][j] = 0
                return is_eq_mat

        ell_mat = np.array([[self.ellipsoid(np.eye(3)), self.ellipsoid(1.0001*np.eye(3)),
                             self.ellipsoid(np.eye(2))],
                            [self.ellipsoid(np.array([[0, ], [1, ], [2, ]]), np.ones((3, 3))),
                             self.ellipsoid(1.0000000001 * np.eye(3)), self.ellipsoid(np.eye(3))],
                            [self.ellipsoid(np.eye(4)),
                             self.ellipsoid(np.array([[0, ], [1, ], [2, ]]), np.ones((3, 3))),
                             self.ellipsoid(np.eye(5))]])
        copied_ell_mat = ell_mat.copy()
        assert np.all(is_equal(copied_ell_mat, ell_mat))
        first_cut_ell_mat = ell_mat[0:2, 0:2]
        second_cut_ell_mat = ell_mat[1:3, 1:3]
        third_cut_ell_mat = ell_mat[0:2, 1:3]
        is_equal_mat = is_equal(first_cut_ell_mat, second_cut_ell_mat)
        is_ok_mat = is_equal_mat == np.array([[1, 0], [1, 0]])
        assert np.all(is_ok_mat)

        is_equal_mat = is_equal(first_cut_ell_mat, third_cut_ell_mat)
        is_ok_mat = is_equal_mat == np.array([[0, 0], [0, 1]])
        assert np.all(is_ok_mat)
