from ellipy.gras.geom.ell.ell import *
import numpy as np


class TestEll:
    def test_rho_mat(self):
        max_tol = 1e-14
        abs_tol = 1e-7
        q_mat = np.array([[49, 4], [4, 1]])
        c_vec = np.array([[1], [0]])
        dirs_mat = np.array([[1, 0], [0, 1]])
        sup_arr, bp_mat = rho_mat(q_mat, dirs_mat, abs_tol, c_vec)
        is_ok = np.bitwise_and((np.abs(sup_arr - np.array([[8, 1]])) < max_tol).transpose(),
                               ((np.abs(bp_mat[:, 1] - np.array([[5, 1]]))) < max_tol).transpose())
        assert np.array_equal(is_ok, np.array([[True], [True]]))

        q_2_mat = np.eye(3)
        c_2_vec = np.array([[1], [0], [0]])
        dirs_mat = np.array([[1], [0], [0]])
        sup_arr, bp_mat = rho_mat(q_2_mat, dirs_mat, abs_tol, c_2_vec)
        is_ok = np.bitwise_and((np.abs(sup_arr - 2) < max_tol).transpose(),
                               np.abs(bp_mat[0] - 2) < max_tol)
        assert np.array_equal(np.array([[True]]), is_ok)




