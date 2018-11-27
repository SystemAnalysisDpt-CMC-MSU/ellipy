from ellipy.gras.geom.ell.ell import *
import numpy as np
from numpy.linalg import norm, inv
from scipy.linalg import hilbert as hilb
from scipy.linalg import invhilbert as invhilb
from ellipy.gen.common.common import throw_error, is_numeric


class TestEll:
    def test_inv_mat(self):
        __EPS = 1e-15
        tmp = 1.
        for x in range(2, 12):
            tmp = tmp * (norm(invhilb(x) - inv_mat(hilb(x))) -
                         norm(invhilb(x) - inv(hilb(x))))
        is_ok = abs(tmp) < __EPS
        assert np.array_equal(True, is_ok)

    def test_quad_mat(self):
        __MAX_TOL = 1e-10
        q_mat = np.array([[2, 5, 7], [6, 3, 4], [5, -2, -3]])
        x_vec = np.array([7, 8, 9]).T
        c_vec = np.array([1, 0, 1])
        calc_mode = 'plain'
        __ANALYTICAL_RESULT_1 = 1304
        __ANALYTICAL_RESULT_2 = 1563
        __ANALYTICAL_RESULT_3 = -364

        def check(analytical_result, calc_mode, c_vec):
            quad_res = quad_mat(q_mat, x_vec, c_vec, calc_mode)
            is_ok = (abs(quad_res - analytical_result) < __MAX_TOL)
            assert np.array_equal(True, is_ok)

        check(__ANALYTICAL_RESULT_1, calc_mode, c_vec)
        c_vec = 0
        check(__ANALYTICAL_RESULT_2, calc_mode, c_vec)
        calc_mode = 'InvAdv'
        c_vec = np.array([1, 0, 1])
        check(__ANALYTICAL_RESULT_3, calc_mode, c_vec)
        calc_mode = 'INV'
        check(__ANALYTICAL_RESULT_3, calc_mode, c_vec)

    def test_quad_mat_negative(self):
        pass

    def test_rho_mat(self):
        __MAX_TOL = 1e-14
        __ABS_TOL = 1e-7
        q_mat = np.array([[49, 4], [4, 1]])
        c_vec = np.array([[1], [0]])
        dirs_mat = np.array([[1, 0], [0, 1]])
        sup_arr, bp_mat = rho_mat(q_mat, dirs_mat, __ABS_TOL, c_vec)
        is_ok = np.bitwise_and((np.abs(sup_arr - np.array([[8, 1]])) < __MAX_TOL).T,
                               ((np.abs(bp_mat[:, 1] - np.array([[5, 1]]))) < __MAX_TOL).T).flatten()
        assert np.array_equal(is_ok, np.array([True, True]))

        q_2_mat = np.eye(3)
        c_2_vec = np.array([[1], [0], [0]])
        dirs_mat = np.array([[1], [0], [0]])
        sup_arr, bp_mat = rho_mat(q_2_mat, dirs_mat, __ABS_TOL, c_2_vec)
        is_ok = np.bitwise_and((np.abs(sup_arr - 2) < __MAX_TOL).transpose(),
                               np.abs(bp_mat[0] - 2) < __MAX_TOL).flatten()[0]
        assert np.array_equal(True, is_ok)

    def test_ell_volume(self):
        __MAX_TOL = 1e-14
        e_vec = np.array([1., 2., 3.])
        q_mat = np.diag(e_vec)
        res_vol = ell_volume(q_mat)
        exp_vol = np.sqrt(np.prod(e_vec)) * np.pi * (4 / 3)
        is_ok = np.abs(exp_vol - res_vol) < __MAX_TOL
        assert is_ok
