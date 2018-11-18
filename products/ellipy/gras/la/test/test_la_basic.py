from ellipy.gras.la.la import *
import pytest
import numpy as np


class TestLaBasic:

    def test_try_treat_as_real(self):
        __ERROR_MSG = 'Incorrect work a try_treat_as_real function'
        real_mat = np.random.rand(3, 3)
        imag_mat = np.eye(3, dtype=np.float64) * (np.finfo(float).eps / 2.0) * 1.0j
        imag_bad_mat = np.eye(3, dtype=np.float64) * (np.finfo(float).eps * 10.0) * 1.0j  # ok<NASGU>
        null_mat = np.zeros((3, 3), dtype=np.float64)
        gib_vec = np.array([0, np.finfo(float).eps * 1.0j / 2.0, 0.0], dtype=complex)
        bad_vec = np.array([1.0, 0.0, 4.0 * 1.0j], dtype=complex)  # ok<NASGU>
        null_vec = np.array([0, 0, 0], dtype=np.float64)

        assert np.array_equal(try_treat_as_real(real_mat), real_mat), __ERROR_MSG
        assert np.array_equal(try_treat_as_real(imag_mat), null_mat), __ERROR_MSG
        assert np.array_equal(try_treat_as_real(gib_vec), null_vec), __ERROR_MSG

        with pytest.raises(Exception) as e:
            try_treat_as_real(imag_bad_mat)
        assert 'wrongInput:inp_mat' in str(e.value)
        with pytest.raises(Exception) as e:
            try_treat_as_real(bad_vec)
        assert 'wrongInput:inp_mat' in str(e.value)

    def test_reg_pos_def_mat(self):
        __REG_TOL = 1e-5
        __ABS_TOL = 1e-8

        # regularize zero matrix
        mat_dim = 10
        zero_mat = np.zeros((mat_dim, mat_dim), dtype=np.float64)
        exp_reg_zero_mat = __REG_TOL * np.eye(mat_dim, dtype=np.float64)
        reg_zero_mat = reg_pos_def_mat(zero_mat, __REG_TOL)
        is_equal = linalg.norm(reg_zero_mat - exp_reg_zero_mat, 2) <= __ABS_TOL
        assert is_equal

        # more complex test
        sh_mat = np.array([[4, 4, 14], [4, 4, 14], [14, 14, 78]], dtype=np.float64)
        is_ok = is_mat_pos_def(sh_mat, __ABS_TOL)
        assert not is_ok
        sh_mat = reg_pos_def_mat(sh_mat, __REG_TOL)
        is_ok = is_mat_pos_def(sh_mat, __ABS_TOL)
        assert not is_ok

        # test small imaginary part
        beg_mat = np.array([[4, 4, 14], [4, 4, 14], [14, 14, 78]], dtype=np.float64)
        imag_mat = reg_pos_def_mat(beg_mat, __REG_TOL + 1.0j * np.finfo(float).eps / 10.0)
        assert np.array_equal(imag_mat, sh_mat), 'Incorrect work reg_pos_def_mat function'

        # negative tests
        wrong_rel_tol = -__REG_TOL  # ok<NASGU>
        with pytest.raises(Exception) as e:
            reg_pos_def_mat(zero_mat, wrong_rel_tol)
        assert 'wrongInput:reg_tol' in str(e.value)

        wrong_rel_tol = np.array([__REG_TOL, __REG_TOL], dtype=np.float64)  # ok<NASGU>
        with pytest.raises(Exception) as e:
            reg_pos_def_mat(zero_mat, wrong_rel_tol)
        assert 'wrongInput:reg_tol' in str(e.value)

        wrong_rel_tol = __REG_TOL + 1.0j * np.finfo(float).eps * 2.0  # ok<NASGU>
        with pytest.raises(Exception) as e:
            reg_pos_def_mat(zero_mat, wrong_rel_tol)
        assert 'wrongInput:inp_mat' in str(e.value)

        non_square_mat = np.zeros((2, 3), dtype=np.float64)  # ok<NASGU>
        with pytest.raises(Exception) as e:
            reg_pos_def_mat(non_square_mat, __REG_TOL)
        assert 'wrongInput:inp_mat' in str(e.value)
