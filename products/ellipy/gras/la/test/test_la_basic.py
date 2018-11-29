from ellipy.gras.la.la import *
from ellipy.gras.gen.gen import *
import numpy as np
import pytest
from ellipy.elltool.conf.properties.Properties import Properties
import scipy.io
import os


class TestLaBasic:

    def test_is_mat_not_deg(self):
        def check(inp_mat, abs_tol, is_ok):
            test_is_ok = is_mat_not_deg(inp_mat, abs_tol)
            assert test_is_ok == is_ok
        q_mat = np.array([[-55, 165], [-20, 60]])
        check(q_mat, 1e-15, False)
        check(q_mat, 1e-17, True)


    def test_sqrt_m_compare(self):
        def check(inp_mat_f, l_tol, r_tol, is_ex_ok):
            is_ok = np.array_equal(sqrtm_pos(inp_mat_f, l_tol), sqrtm_pos(inp_mat_f, r_tol))
            assert is_ok == is_ex_ok

        inp_mat = np.eye(2)
        check(inp_mat, 1.5, 0, True)
        check(inp_mat, 1.5, 2, True)
        check(inp_mat, 1.5, 1, True)
        inp_mat = np.diag([1, -1])
        check(inp_mat, 1.5, 1, True)
        check(inp_mat, 1.5, 1.5, True)

    def test_sqrt_m_simple(self):
        def check_is_pos(eig_vec, is_pos_exp, *args):
            inp_mat_f = np.diag(eig_vec)
            inp_pos = is_mat_pos_def(inp_mat_f, abs_tol, *args)
            assert inp_pos == is_pos_exp

            is_not_neg = is_mat_pos_def(inp_mat_f, abs_tol, True)
            if is_not_neg:
                assert np.all(np.isreal(sqrtm_pos(inp_mat_f, abs_tol)))
                sqrt_vec = sqrt_pos(np.array([eig_vec]), abs_tol)
                exp_sqrt_vec = np.array([sqrt_pos(np.array(x), abs_tol) for x in np.array([eig_vec])])
                assert np.all(sqrt_vec == exp_sqrt_vec)
                assert np.all(np.isreal(sqrt_vec))
            else:
                with pytest.raises(Exception) as s:
                    sqrtm_pos(inp_mat_f, abs_tol)
                assert 'wrongInput:notPosSemDef' in str(s.value)
                with pytest.raises(Exception) as s:
                    sqrt_pos(np.array([eig_vec]), abs_tol)
                assert 'wrongInput:negativeInput' in str(s.value)

        with pytest.raises(Exception) as e:
            sqrtm_pos(np.eye(2), -1)
        assert 'wrongInput:abs_tolNegative' in str(e.value)

        assert np.all(np.isreal(sqrtm_pos(np.diag([0, -0.001]), 0.001)))

        min_eig_val = -0.001
        abs_tol = 0.001
        assert np.all(np.isreal(sqrtm_pos(np.diag([0, min_eig_val]), abs_tol)))

        inp_mat = np.diag([0, 2 * min_eig_val])
        with pytest.raises(Exception) as e:
            sqrtm_pos(inp_mat, abs_tol)
        assert 'wrongInput:notPosSemDef' in str(e.value)

        check_is_pos([0, 2 * min_eig_val], False)
        check_is_pos([0, min_eig_val], False)
        check_is_pos([0, min_eig_val], True, True)
        check_is_pos([10, -2*min_eig_val], True, True)
        check_is_pos([abs_tol, -2*min_eig_val], True, True)

    def test_sqrt_m(self):
        __MAX_TOL = 1e-6
        n_dim = 100
        test_mat = np.eye(n_dim)
        sqrt_mat = sqrtm_pos(test_mat, __MAX_TOL)
        assert np.array_equal(test_mat, sqrt_mat)
        sqrt_mat = sqrtm_pos(test_mat)
        assert np.array_equal(test_mat, sqrt_mat)

        n_dim = [1, 100]
        test_mat = np.diag(n_dim)
        sqrt_mat = sqrtm_pos(test_mat, __MAX_TOL)
        assert np.array_equal(sqrt_pos(test_mat), sqrt_mat)
        sqrt_mat = sqrtm_pos(test_mat)
        assert np.array_equal(sqrt_pos(test_mat), sqrt_mat)

        test_mat = np.array([[2, 1], [1, 2]])
        v_mat = np.array([[-1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
        d_mat = np.diag([1, np.sqrt(3)])
        sqrt_test_mat = v_mat @ d_mat @ v_mat.T
        sqrt_mat = sqrtm_pos(test_mat, __MAX_TOL)
        assert np.array_equal(sqrt_test_mat, sqrt_mat)
        sqrt_mat = sqrtm_pos(test_mat)
        assert np.array_equal(sqrt_test_mat, sqrt_mat)

        test_mat = np.array([[5, -4, 1], [-4, 6, -4], [1, -4, 5]])
        sqrt_test_mat = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        sqrt_mat = sqrtm_pos(test_mat, __MAX_TOL)
        assert np.linalg.norm(sqrtm_pos(sqrt_test_mat, __MAX_TOL) - sqrtm_pos(sqrt_mat, __MAX_TOL)) < __MAX_TOL

        loaded_info = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testSqrtm1_inp.mat'))
        mat = loaded_info['testMat']
        sqrt_mat = sqrtm_pos(mat, __MAX_TOL)
        assert np.linalg.norm(sqrtm_pos(mat, __MAX_TOL) - sqrtm_pos(sqrt_mat@sqrt_mat.T, __MAX_TOL)) < __MAX_TOL

        loaded_info = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testSqrtm2_inp.mat'))
        mat = loaded_info['testMat']
        sqrt_mat = sqrtm_pos(mat, __MAX_TOL)
        assert np.linalg.norm(sqrtm_pos(mat, __MAX_TOL) - sqrtm_pos(sqrt_mat @ sqrt_mat.T)) < __MAX_TOL

        test_1_mat = np.eye(2)
        test_2_sqrt_mat = np.eye(2) + 1.01 * __MAX_TOL
        test_2_mat = test_2_sqrt_mat @ test_2_sqrt_mat.transpose()
        assert np.linalg.norm(sqrtm_pos(test_1_mat, __MAX_TOL) - sqrtm_pos(test_2_mat, __MAX_TOL)) > __MAX_TOL

        test_1_mat = np.eye(2)
        test_2_sqrt_mat = np.eye(2) + 0.5 * __MAX_TOL
        test_2_mat = test_2_sqrt_mat@test_2_sqrt_mat.T
        assert np.linalg.norm(sqrtm_pos(test_1_mat, __MAX_TOL) - sqrtm_pos(test_2_mat, __MAX_TOL)) < __MAX_TOL

        test_mat = np.array([[1, 0], [0, -1]])
        with pytest.raises(Exception) as e:
            sqrt_pos(np.array(test_mat), __MAX_TOL)
        assert 'wrongInput' in str(e.value)

        with pytest.raises(Exception) as e:
            sqrtm_pos(np.array([[-11, 30], [-10, 24]]))
        assert 'wrongInput:nonSymmMat' in str(e.value)

    def test_is_mat_pos_simple(self):
        is_ok = not is_mat_pos_def(np.zeros((2, 2)), 1e-7)
        assert is_ok
        is_ok = is_mat_pos_def(np.zeros((2, 2)), 1e-7, True)
        assert is_ok
        is_ok = not is_mat_pos_def(np.zeros((2, 2)), 1e-7, False)
        assert is_ok
        is_ok = not is_mat_pos_def(np.zeros((2, 2)))
        assert is_ok
        is_ok = is_mat_pos_def(np.zeros((2, 2)), 0, True)
        assert is_ok

    def test_is_mat_pos_and_pos_sem_def(self):
        abs_tol = Properties.get_abs_tol()

        def check(f):
            assert f(np.array([[1]]), abs_tol)

            test_mat_check = np.random.rand(10, 10)
            test_mat_check = test_mat_check.T @ test_mat_check
            _, v_mat = np.linalg.eigh(test_mat_check)
            d_mat = np.diag(np.arange(1, 11))
            test_mat_check = v_mat.T @ d_mat @ v_mat
            test_mat_check = 0.5 * (test_mat_check.T + test_mat_check)
            is_ok = f(test_mat_check, abs_tol)
            assert is_ok

        def check_mult_times():
            test_mat_check = np.random.rand(5, 5)
            test_mat_check = test_mat_check.T @ test_mat_check
            _, v_mat = np.linalg.eigh(test_mat_check)
            d_mat = np.diag(np.arange(1, 6))
            test_mat_check = v_mat.T @ d_mat @ v_mat
            test_mat_check = -0.5 * (test_mat_check.T + test_mat_check)
            is_false = is_mat_pos_def(test_mat_check, abs_tol, True)
            assert is_false is False
            with pytest.raises(Exception) as s:
                sqrtm_pos(test_mat_check, abs_tol)
            assert 'wrongInput:notPosSemDef' in str(s.value)

        def check_determ(orth3mat_check, diag_vec_check, is_true, is_sem_pos_def: bool = None):
            test_mat_check = orth3mat_check.T @ np.diag(diag_vec_check) @ orth3mat_check
            test_mat_check = 0.5 * (test_mat_check + test_mat_check.T)
            if is_sem_pos_def is None:
                is_ok = is_mat_pos_def(test_mat_check, abs_tol)
            else:
                is_ok = is_mat_pos_def(test_mat_check, abs_tol, is_sem_pos_def)
            assert is_true == is_ok

        def f_is_mat_pos_sem_def(q_mat, abs_tol_test):
            return is_mat_pos_def(q_mat, abs_tol_test, True)

        def f_is_mat_pos_def(q_mat, abs_tol_test):
            return is_mat_pos_def(q_mat, abs_tol_test, False)

        check(is_mat_pos_def)
        check(f_is_mat_pos_sem_def)
        check(f_is_mat_pos_def)

        test_mat = np.random.rand(10, 10)
        test_mat = test_mat.T @ test_mat
        test_mat = 0.5 * (test_mat + test_mat.T)
        assert f_is_mat_pos_sem_def(test_mat.T @ test_mat, abs_tol)

        test_mat = np.array([[1, 5], [5, 25]])
        assert not is_mat_pos_def(test_mat, abs_tol)
        assert f_is_mat_pos_sem_def(test_mat, abs_tol)
        assert not f_is_mat_pos_def(test_mat, abs_tol)

        assert is_mat_pos_def(np.eye(3))

        with pytest.raises(Exception) as e:
            is_mat_pos_def(np.eye(3, 5), abs_tol)
        assert 'wrongInput:nonSquareMat' in str(e.value)

        with pytest.raises(Exception) as e:
            is_mat_pos_def(np.array([[1, -1], [1, 1]]), abs_tol)
        assert 'wrongInput:nonSymmMat' in str(e.value)

        n_times = 50
        for i_time in range(n_times):
            check_mult_times()

        is_pos_or_sem_def = True
        orth3mat = np.array([[-0.206734513608356, -0.439770172956299, 0.873992583413099],
                             [0.763234588112547, 0.486418086920488, 0.425288617559045],
                             [-0.612155049306781, 0.754983204908957, 0.23508840021067]])
        diag_vec = [1, 2, 3]
        check_determ(orth3mat, diag_vec, is_pos_or_sem_def)

        diag_vec = [1, -1, 3]
        check_determ(orth3mat, diag_vec, not is_pos_or_sem_def)

        diag_vec = [0, 1, 1]
        check_determ(orth3mat, diag_vec, not is_pos_or_sem_def)

        diag_vec = [0, 1, 2]
        check_determ(orth3mat, diag_vec, is_pos_or_sem_def, True)

        diag_vec = [0, -1, 2]
        check_determ(orth3mat, diag_vec, not is_pos_or_sem_def, True)

        diag_vec = [-1, 1, -2]
        check_determ(orth3mat, diag_vec, not is_pos_or_sem_def, False)
        check_determ(orth3mat, diag_vec, not is_pos_or_sem_def, True)

    def test_try_treat_as_real(self):
        __ERROR_MSG = 'Incorrect work a try_treat_as_real function'
        real_mat = np.random.rand(3, 3)
        imag_mat = np.eye(3, dtype=np.float64) * (np.finfo(float).eps / 2.0) * 1.0j
        imag_bad_mat = np.eye(3, dtype=np.float64) * (np.finfo(float).eps * 10.0) * 1.0j
        null_mat = np.zeros((3, 3), dtype=np.float64)
        gib_vec = np.array([0, np.finfo(float).eps * 1.0j / 2.0, 0.0], dtype=complex)
        bad_vec = np.array([1.0, 0.0, 4.0 * 1.0j], dtype=complex)
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

    def test_reg_mat(self):
        __ERROR_MSG = 'Incorrect work a reg_mat function'
        diag_mat = np.eye(3, dtype=np.float64) * 3.0
        par_mat = 5.0 * np.eye(3, dtype=np.float64)
        assert np.array_equal(reg_mat(diag_mat, 5.0), par_mat), __ERROR_MSG
        ress_mat = reg_mat(np.array([[3.0]]), 1.0)
        assert np.array_equal(ress_mat, np.array([[3.]])), __ERROR_MSG
        ress_mat = reg_mat(np.array([[3.0]]), 1.0 + np.finfo(float).eps * 1.0j / 5.0)
        assert np.array_equal(ress_mat, np.array([[3.0]])), __ERROR_MSG
        with pytest.raises(Exception) as e:
            reg_mat(diag_mat, -1.)
        assert 'wrongInput:reg_tol' in str(e.value)
        with pytest.raises(Exception) as e:
            reg_mat(diag_mat, 2.0 + np.finfo(float).eps * 1.0j * 2.0)
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
        assert is_ok

        # test small imaginary part
        beg_mat = np.array([[4, 4, 14], [4, 4, 14], [14, 14, 78]], dtype=np.float64)
        imag_mat = reg_pos_def_mat(beg_mat, __REG_TOL + 1.0j * np.finfo(float).eps / 10.0)
        assert np.array_equal(imag_mat, sh_mat), 'Incorrect work reg_pos_def_mat function'

        # negative tests
        wrong_rel_tol = -__REG_TOL
        with pytest.raises(Exception) as e:
            reg_pos_def_mat(zero_mat, wrong_rel_tol)
        assert 'wrongInput:reg_tol' in str(e.value)

        wrong_rel_tol = np.array([__REG_TOL, __REG_TOL], dtype=np.float64)
        with pytest.raises(Exception) as e:
            # noinspection PyTypeChecker
            reg_pos_def_mat(zero_mat, wrong_rel_tol)
        assert 'wrongInput:reg_tol' in str(e.value)

        wrong_rel_tol = __REG_TOL + 1.0j * np.finfo(float).eps * 2.0
        with pytest.raises(Exception) as e:
            reg_pos_def_mat(zero_mat, wrong_rel_tol)
        assert 'wrongInput:inp_mat' in str(e.value)

        non_square_mat = np.zeros((2, 3), dtype=np.float64)
        with pytest.raises(Exception) as e:
            reg_pos_def_mat(non_square_mat, __REG_TOL)
        assert 'wrongInput:nonSquareMat' in str(e.value)
