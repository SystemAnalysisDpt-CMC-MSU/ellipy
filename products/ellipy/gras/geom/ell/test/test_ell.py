from ellipy.gras.geom.ell.ell import *
from numpy.linalg import norm, inv
from scipy.linalg import hilbert as hilb
from scipy.linalg import invhilbert as invhilb
from ellipy.gen.common.common import *
import pytest


class TestEll:
    def test_inv_mat(self):
        norm_diff_vec = np.zeros(10, dtype=np.float64)
        for x in range(2, 12):
            norm_diff_vec[x - 2] = (norm(invhilb(x) - inv_mat(hilb(x))) -
                                    norm(invhilb(x) - inv(hilb(x))))
        is_ok = np.prod(norm_diff_vec) == 0
        assert is_ok

    def test_quad_mat(self):
        __MAX_TOL = 1e-10
        q_mat = np.array([[2, 5, 7],
                          [6, 3, 4],
                          [5, -2, -3]], dtype=np.int64)
        x_vec = np.array([7, 8, 9], dtype=np.int64).T
        c_vec = np.array([1, 0, 1], dtype=np.int64)
        calc_mode = 'plain'
        __ANALYTICAL_RESULT_1 = 1304
        __ANALYTICAL_RESULT_2 = 1563
        __ANALYTICAL_RESULT_3 = -364

        def check(analytical_result, mode, c_vector):
            quad_res = quad_mat(q_mat, x_vec, c_vector, mode)
            is_ok = abs(quad_res - analytical_result) < __MAX_TOL
            assert is_ok

        check(__ANALYTICAL_RESULT_1, calc_mode, c_vec)
        c_vec = 0
        check(__ANALYTICAL_RESULT_2, calc_mode, c_vec)
        calc_mode = 'InvAdv'
        c_vec = np.array([1, 0, 1], dtype=np.int64)
        check(__ANALYTICAL_RESULT_3, calc_mode, c_vec)
        calc_mode = 'INV'
        check(__ANALYTICAL_RESULT_3, calc_mode, c_vec)

    def test_quad_mat_negative(self):
        q_mat_square = np.array([[1, 0],
                                 [0, 1]], dtype=np.int64)
        q_mat_not_square = np.array([1, 0], dtype=np.int64)
        x_vec_good_dim = np.array([3, 2], dtype=np.int64)
        x_vec_bad_dim = np.array([1, 5, 10], dtype=np.int64)
        c_vec_good_dim = np.array([1, 1], dtype=np.int64)
        c_vec_bad_dim = np.array([1, 3, 7], dtype=np.int64)
        mode = 'plain'

        with pytest.raises(Exception) as e:
            quad_mat(q_mat_not_square, x_vec_good_dim, c_vec_good_dim, mode)
        assert 'wrongInput' in str(e.value)

        with pytest.raises(Exception) as e:
            quad_mat(q_mat_square, x_vec_bad_dim, c_vec_good_dim, mode)
        assert 'wrongInput' in str(e.value)

        with pytest.raises(Exception) as e:
            quad_mat(q_mat_square, x_vec_good_dim, c_vec_bad_dim, mode)
        assert 'wrongInput' in str(e.value)

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
