from ellipy.gras.la.la import *
from ellipy.gras.geom.tri.tri import *
from ellipy.gras.geom.sup.sup import *
from ellipy.gras.geom.geom import circle_part
import numpy as np
import pytest
import scipy.io
import numpy.matlib
import scipy as sc
import os


class TestSup:
    def test_sup_2_boundary_2(self):
        q_mat = np.diag([1, 2])
        s_mat = orth_transl(np.array([1, 0]), np.array([1, 1]))
        q_mat = s_mat.T @ q_mat @ s_mat
        dir_mat = circle_part(100)
        self.aux_test_sup_boundary(sup_2_boundary_2, sup_2_boundary_2, q_mat, dir_mat)

    def test_sup_2_boundary_3(self):
        q_mat = np.diag([1, 2, 3])
        s_mat = orth_transl(np.array([1, 0, 0]), np.array([1, 1, 1]))
        q_mat = s_mat.T @ q_mat @ s_mat
        dir_mat, face_mat = sphere_tri(6)
        self.aux_test_sup_boundary(sup_2_boundary_3, sup_2_boundary_3, q_mat, dir_mat, face_mat)

    @staticmethod
    def aux_test_sup_boundary(f_boundary, f_check_boundary, q_mat, dir_mat, *args):
        __MAX_NORM = 1 + 1e-3
        __MIN_NORM = 1
        __MAX_TOL = 1e-14
        sup_vec = np.sqrt(np.sum((dir_mat @ q_mat) * dir_mat, 1).T)
        x_mat = f_boundary(dir_mat, sup_vec, *args)
        x_exp_mat = f_check_boundary(dir_mat, sup_vec, *args)
        real_tol = np.max(np.sqrt(np.sum(((x_mat - x_exp_mat) ** 2).T, 1)))
        assert real_tol <= __MAX_TOL
        y_mat = np.linalg.lstsq(sc.linalg.sqrtm(q_mat).T, x_mat.T, - 1)[0].T
        n_vec = np.sqrt(np.sum(y_mat ** 2, 1))
        assert max(n_vec) <= __MAX_NORM
        assert min(n_vec) >= __MIN_NORM

    def test_sup_geom_diff_2d(self):
        __N_DIRS = 200
        __EXP_TOL = 1e-15
        __EXP_MAX = 0.612493409916315
        __EXP_MIN = 0.105572809000084

        def rho(q_mat: np.ndarray, dir_mat: np.ndarray):
            rho_vec = np.sqrt(np.sum((q_mat @ dir_mat) * dir_mat, axis=0))
            return rho_vec

        l_mat = circle_part(__N_DIRS).T

        q1_mat = np.diag([1, 2])
        q2_mat = np.diag([0.8, 0.1])
        rho1_vec = rho(q1_mat, l_mat)
        rho2_vec = rho(q2_mat, l_mat)
        rho_diff_vec = sup_geom_diff_2d(rho1_vec, rho2_vec, l_mat)
        n_dirs_shift = np.long(np.fix(__N_DIRS * 0.5))
        max_period_tol = np.max(np.abs(np.roll(rho_diff_vec,
                                               (1, n_dirs_shift - 1)) - rho_diff_vec))
        assert max_period_tol <= __EXP_TOL
        assert np.abs(__EXP_MAX - np.max(rho_diff_vec)) <= __EXP_TOL
        assert np.abs(__EXP_MIN - np.min(rho_diff_vec)) <= __EXP_TOL

    def test_sup_geom_diff_2d_negative(self):
        __N_DIRS = 200

        l_mat = circle_part(__N_DIRS).T
        rho1_vec = np.ones(__N_DIRS)
        rho2_vec = np.ones(__N_DIRS) * 0.5

        with pytest.raises(Exception) as e:
            sup_geom_diff_2d(rho2_vec, rho1_vec, l_mat)
        assert 'wrongInput:rho_diff_vec' in str(e.value)

        with pytest.raises(Exception) as e:
            sup_geom_diff_2d(rho2_vec, rho1_vec, l_mat.T)
        assert 'wrongInput:rho1_vec,l_mat' in str(e.value)

        with pytest.raises(Exception) as e:
            sup_geom_diff_2d(rho2_vec, np.matlib.repmat(rho1_vec, 2, 1), l_mat)
        assert 'wrongInput:rho2_vec' in str(e.value)

        with pytest.raises(Exception) as e:
            sup_geom_diff_2d(np.matlib.repmat(rho2_vec, 2, 1), rho1_vec, l_mat.T)
        assert 'wrongInput:rho1_vec' in str(e.value)

    def test_sup_geom_diff_3d(self):
        __ABS_TOL = 1e-10
        __POINTS_NUMBER = 200
        __TEST_DATA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        __SUPP1_MAT = \
            scipy.io.loadmat(os.path.join(__TEST_DATA_ROOT_DIR, 'supp1_mat_data.mat'))['supp1Mat']
        __SUPP2_MAT = \
            scipy.io.loadmat(os.path.join(__TEST_DATA_ROOT_DIR, 'supp2_mat_data.mat'))['supp2Mat']
        __SEC_BOUND_MAT = \
            scipy.io.loadmat(os.path.join(__TEST_DATA_ROOT_DIR, 'sec_bound_mat_data.mat'))['secBoundMat']

        l_grid_mat, _ = sphere_tri(3)
        supp1_mat = __SUPP1_MAT[0]
        supp2_mat = __SUPP2_MAT[0]
        rho_diff_vec = sup_geom_diff_3d(supp1_mat, supp2_mat, l_grid_mat.T)
        l_grid2_mat = np.diag([-1, -1, 1]) @ l_grid_mat.T
        rho_diff2_vec = sup_geom_diff_3d(supp1_mat, supp2_mat, l_grid2_mat)

        assert np.array_equal(np.abs(rho_diff_vec - rho_diff2_vec) < __ABS_TOL, np.ones(rho_diff_vec.shape[0]))

        fir_bound_mat = circle_part(__POINTS_NUMBER).T
        sec_bound_mat = __SEC_BOUND_MAT
        third_bound_mat = np.concatenate((numpy.matlib.repmat(fir_bound_mat, 1, 10),
                                          np.floor(np.expand_dims(np.arange(0, 2000), axis=0) /
                                                   __POINTS_NUMBER)), axis=0)
        forth_bound_mat = np.concatenate((np.matlib.repmat(sec_bound_mat, 1, 10),
                                          np.floor(np.expand_dims(np.arange(0, 2000), axis=0) /
                                                   __POINTS_NUMBER) / 10), axis=0)
        l_grid_mat = np.concatenate((fir_bound_mat,
                                     np.expand_dims(np.zeros(__POINTS_NUMBER), axis=0)), axis=0).T
        l_grid_mat = np.concatenate((l_grid_mat,
                                     np.expand_dims(np.array([.5, .5, .5], dtype=np.float64), axis=0)), axis=0)
        sup1_vec = np.amax(fir_bound_mat.T @ fir_bound_mat, axis=1)
        sup2_vec = np.amax(fir_bound_mat.T @ sec_bound_mat, axis=1)
        sup3_vec = np.max(l_grid_mat @ third_bound_mat, axis=1)
        sup4_vec = np.max(l_grid_mat @ forth_bound_mat, axis=1)

        rho_diff_vec = sup_geom_diff_2d(sup1_vec.T, sup2_vec.T, fir_bound_mat)
        rho_diff2_vec = sup_geom_diff_3d(sup3_vec.T, sup4_vec.T, l_grid_mat.T)

        assert np.array_equal(np.abs(rho_diff2_vec[0: rho_diff2_vec.shape[0] - 1] -
                                     rho_diff_vec) < __ABS_TOL, np.ones(rho_diff_vec.shape[0]))
