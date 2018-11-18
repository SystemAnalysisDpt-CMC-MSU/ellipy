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
