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
        dst_vec = np.array([[1], [2 + 1.0j * np.finfo(float).eps/2]], dtype=complex)

        oimag_mat = orth_transl(src_vec, dst_vec)
        dst_vec = np.array([[1], [2]], dtype=np.float64)
        oreal_mat = orth_transl(src_vec, dst_vec)
        assert np.array_equal(oimag_mat, oreal_mat), 'Incorrect work orth_transl function'

    def test_ml_orth_transl(self):
        self.aux_test_qorth(ml_orth_transl, orth_transl)

    def aux_test_qorth(self, f_handle, f_handle_single) -> float:
        def check(o_mat, src_vec, dst_exp_vec):
            self.aux_check_orth(o_mat, src_vec, dst_exp_vec, '{}'.format(f_handle))
            o_exp_mat = f_handle_single(src_vec, dst_exp_vec)
            real_tol = np.max(np.max(np.abs(o_mat - o_exp_mat)))
            is_pos = real_tol <= self.__MAX_TOL
            assert is_pos, 'when comparing {} and {} real tol {}>{}'\
                .format(f_handle, f_handle_single, real_tol, self.__MAX_TOL)

        __N_ELEMS = 1000
        src_mat = self.__SRC_TL_MAT
        dst_mat = self.__DST_TL_MAT
        dst_array = np.tile(dst_mat[:, :, np.newaxis], (1, 1, __N_ELEMS))
        n_vecs = np.size(src_mat, 1)
        t_start = timer()
        o_array = f_handle(src_mat, dst_array)
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
        assert np.size(o_mat, 0) == np.size(o_mat, 1)
        self.aux_check_eye(o_mat.T @ o_mat, 'o_mat^t*o_mat', func_name)
        self.aux_check_eye(o_mat @ o_mat.T, 'o_mat*o_mat^t', func_name)

    def aux_check_eye(self, e_mat, msg_str, func_name):
        n_dims = np.size(e_mat, 0)
        real_tol = np.max(np.max(np.abs(e_mat - np.eye(n_dims)), axis=0))
        is_pos = real_tol <= self.__MAX_TOL
        assert is_pos, 'real tol for {}=I check of {} is {}>{}'.format(msg_str, func_name, real_tol, self.__MAX_TOL)
