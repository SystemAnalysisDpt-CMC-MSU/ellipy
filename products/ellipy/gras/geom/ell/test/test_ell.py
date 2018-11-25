from ellipy.gras.geom.ell.ell import *
import numpy as np


class TestEll:
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
