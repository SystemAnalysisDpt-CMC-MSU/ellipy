from ellipy.gras.la.la import *
import pytest
import numpy as np
from timeit import default_timer as timer


class TestOrthTransl:
    __SRC_TL_MAT = np.array([[0, 3], [1, 1], [0.3, -4], [-2, 1]], dtype=np.float64)
    __DST_TL_MAT = np.array([[-1, -5], [2, 2], [3, 1], [4, 5]], dtype=np.float64)
    __MAX_TOL = 1e-8

    def test_orth_transl(self):
        def check(check_src_vec, check_dst_vec, exp_error_tag):
            with pytest.raises(Exception) as e:
                orth_transl(check_src_vec, check_dst_vec)
            assert exp_error_tag in str(e.value)
            with pytest.raises(Exception) as e:
                    orth_transl_qr(check_src_vec, check_dst_vec)
            assert exp_error_tag in str(e.value)

        src_vec = np.array([0, 0], dtype=np.float64)
        dst_vec = np.array([1, 0], dtype=np.float64)
        check(src_vec, dst_vec, 'wrongInput:src_zero')

        src_vec = np.array([1, 0], dtype=np.float64)
        dst_vec = np.array([0, 0], dtype=np.float64)
        check(src_vec, dst_vec, 'wrongInput:dst_zero')

        src_vec = np.array([1, 0], dtype=np.float64) + 1.0j * np.array([1, 0], dtype=np.float64)
        check(src_vec, dst_vec, 'wrongInput:inp_mat')

        src_vec = np.array([1, 0], dtype=np.float64)
        dst_vec = np.array([1, 0], dtype=np.float64) + 1.0j * np.array([1, 0], dtype=np.float64)
        check(src_vec, dst_vec, 'wrongInput:inp_mat')

        src_vec = np.array([[2], [5]], dtype=np.float64)
        dst_vec = np.array([[1], [2 + 1.0j * np.finfo(float).eps/2]], dtype=np.complex128)

        oimag_mat = orth_transl(src_vec, dst_vec)
        dst_vec = np.array([[1], [2]], dtype=np.float64)
        oreal_mat = orth_transl(src_vec, dst_vec)
        assert np.array_equal(oimag_mat, oreal_mat), 'Incorrect work orth_transl function'

    def test_ml_orth_transl(self):
        self.aux_test_qorth(ml_orth_transl, orth_transl)

    def aux_test_qorth(self, f, f_single) -> float:
        def check(o_mat, src_vec, dst_exp_vec):
            self.aux_check_orth(o_mat, src_vec, dst_exp_vec, '{}'.format(f))
            o_exp_mat = f_single(src_vec, dst_exp_vec)
            real_tol = np.max(np.max(np.abs(o_mat - o_exp_mat)))
            is_pos = real_tol <= self.__MAX_TOL
            assert is_pos, 'when comparing {} and {} real tol {}>{}'\
                .format(f, f_single, real_tol, self.__MAX_TOL)

        __N_ELEMS = 1000
        src_mat = self.__SRC_TL_MAT
        dst_mat = self.__DST_TL_MAT
        dst_array = np.tile(dst_mat[:, :, np.newaxis], (1, 1, __N_ELEMS))
        n_vecs = np.size(src_mat, 1)
        t_start = timer()
        o_array = f(src_mat, dst_array)
        t_elapsed = timer() - t_start
        for i_elem in range(__N_ELEMS):
            for i_vec in range(n_vecs):
                check(o_array[:, :, i_elem, i_vec], src_mat[:, i_vec], dst_array[:, i_vec, i_elem])
        return t_elapsed

    def aux_check_orth(self, o_mat, src_vec, dst_exp_vec, func_name):
        self.aux_check_orth_plain(o_mat, func_name)

        dst_vec = o_mat @ src_vec

        dst_vec = dst_vec / np.linalg.norm(dst_vec)
        dst_exp_vec = dst_exp_vec / np.linalg.norm(dst_exp_vec)

        real_tol = np.max(np.abs(dst_vec - dst_exp_vec))
        is_pos = real_tol <= self.__MAX_TOL
        assert is_pos, 'dst_vec for {} is not close enough to dst_exp_vec, is {}>{}'\
            .format(func_name, real_tol, self.__MAX_TOL)

    def aux_check_orth_plain(self, o_mat, func_name):
        assert (o_mat.ndim <= 2)
        if np.size(o_mat) == 1:
            self.aux_check_eye(o_mat ** 2, 'o_mat^t*o_mat', func_name)
        else:
            assert np.size(o_mat, 0) == np.size(o_mat, 1)
            self.aux_check_eye(o_mat.T @ o_mat, 'o_mat^t*o_mat', func_name)
            self.aux_check_eye(o_mat @ o_mat.T, 'o_mat*o_mat^t', func_name)

    def aux_check_eye(self, e_mat, msg_str, func_name):
        n_dims = np.size(e_mat, 0)
        real_tol = np.max(np.max(np.abs(e_mat - np.eye(n_dims)), axis=0))
        is_pos = real_tol <= self.__MAX_TOL
        assert is_pos, 'real tol for {}=I check of {} is {}>{}'.format(msg_str, func_name, real_tol, self.__MAX_TOL)

    def test_mat_orth(self):
        def check(inp_mat):
            o_mat = mat_orth(inp_mat)
            o_red_mat = math_orth_col(inp_mat)
            assert np.array_equal(o_mat[:, :inp_mat.shape[1]], o_red_mat)
            self.aux_check_orth_plain(o_mat, 'matorth')

        inp_mat = self.__SRC_TL_MAT
        n_cols = inp_mat.shape[1]
        for i_col in range(n_cols):
            check(inp_mat[:, :i_col + 1])

    def test_orth_transl_max(self):
        __N_RANDOM_CASES = 10
        __DIM_VEC = np.array([[1, 2, 3, 5]], dtype=np.int32)
        __ALT_TOL = 1e-10

        def master_check(mas_ch_src_mat, mas_ch_dst_mat):
            def check_metric(f_calc, o_max_mat, o_comp_mat):
                MAX_METRIC_COMP_TOL = 1e-13
                max_val = f_calc(o_max_mat)
                comp_val = f_calc(o_comp_mat)
                is_pos = max_val + MAX_METRIC_COMP_TOL >= comp_val
                assert is_pos, "{} maximization doesn't work, max_val {} < comp_val {}" .format(f_calc, max_val, comp_val)

            def check(f_prod, f_test, f_comp, *args):
                check_src_vec, check_dst_vec, check_a_mat = args
                check_o_mat = f_prod(*args)
                self.aux_check_orth(check_o_mat, check_src_vec, check_dst_vec, '{}'.format(f_prod))

                o_exp_mat = f_test(*args)
                self.aux_check_orth(o_exp_mat, check_src_vec, check_dst_vec, '{}'.format(f_test))
                comp_val = f_comp(check_o_mat, check_a_mat)
                comp_exp_val = f_comp(o_exp_mat, check_a_mat)
                real_tol = np.max(np.abs(comp_val - comp_exp_val))
                is_pos = real_tol <= self.__MAX_TOL
                assert is_pos, 'when comparing {} and {} real tol {}>{}'.format(
                    f_prod, f_test, real_tol, self.__MAX_TOL)

            def calc_trace(inp_o_mat, inp_a_mat):
                return np.trace(inp_o_mat @ inp_a_mat)

            def calc_dir(o_mat):
                if np.size(o_mat) > 1:
                    return np.transpose(dst_max_vec) @ o_mat @ src_max_vec
                else:
                    return o_mat * np.transpose(dst_max_vec) @ src_max_vec

            src_vec = mas_ch_src_mat[:, 0]
            dst_vec = mas_ch_dst_mat[:, 0]

            # Test Hausholder function
            o_mat = orth_transl_haus(src_vec, dst_vec)
            self.aux_check_orth(o_mat, src_vec, dst_vec, orth_transl_haus)

            # Test MAX Trace functions
            n_dims_max_tr = np.size(src_vec)
            a_sqrt_mat = np.random.rand(n_dims_max_tr, n_dims_max_tr)
            a_mat = a_sqrt_mat @ a_sqrt_mat.transpose()
            check(orth_transl_max_tr, orth_transl_max_tr, calc_trace, src_vec, dst_vec, a_mat)

            # Test MAX Dir functions
            src_max_vec = src_mat[:, 1]
            dst_max_vec = dst_mat[:, 1]
            o_max_dir_mat = check(orth_transl_max_dir, orth_transl_max_dir, calc_dir, src_vec, dst_vec, src_max_vec, dst_max_vec)
            o_plain_mat = orth_transl(src_vec, dst_vec)

        master_check(self.__SRC_TL_MAT, self.__DST_TL_MAT)


        for n_dims in __DIM_VEC.flat:
            for i_Test in range(1, __N_RANDOM_CASES + 1):
                src_mat = np.random.rand(n_dims, 2)
                dst_mat = np.random.rand(n_dims, 2)

                master_check(src_mat, dst_mat)

                master_check(src_mat, src_mat)

                dst_alt_mat = src_mat + (np.random.rand(n_dims, 2)) * __ALT_TOL
                master_check(src_mat, dst_alt_mat)

    def test_orth_transl_qr(self):
        __CALC_PRECISION = 1e-10
        __EPS = 1e-17

        def check(src_vec, dst_vec):
            ind = np.flatnonzero(dst_vec)[0]
            o_mat = orth_transl_qr(src_vec, dst_vec)
            got_vec = o_mat @ src_vec
            diff_vec = np.abs(dst_vec / dst_vec[ind] - got_vec / got_vec[ind])
            assert np.all(diff_vec < __CALC_PRECISION)

        check(np.array([1]), np.array([-1]))
        check(np.array([10]), np.array([2]))
        check(np.array([[1], [0]]), np.array([[0], [1]]))
        check(self.__SRC_TL_MAT[:, 0], self.__DST_TL_MAT[:, 0])
        check(self.__SRC_TL_MAT[:, 1], self.__DST_TL_MAT[:, 1])
        o_imag_mat = orth_transl_qr(np.array([[2+1j*__EPS/10], [5]]), np.array([[1], [2+1j*__EPS/2]]))
        o_real_mat = orth_transl_qr(np.array([[2], [5]]), np.array([[1], [2]]))
        assert np.array_equal(o_imag_mat, o_real_mat), 'Incorrect work orth_transl function'
