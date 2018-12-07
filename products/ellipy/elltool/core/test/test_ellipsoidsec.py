from ellipy.elltool.core.core import *
from ellipy.elltool.core.hyperplane.Hyperplane import *
from ellipy.gen.common.common import abs_rel_compare
from typing import Tuple


class TestEllipsoidSecTestCase:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def ell_unitball(self, *args, **kwargs):
        return ell_unitball(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def hyperplane(self, *args, **kwargs):
        return Hyperplane(*args, **kwargs)

    def test_minksum_ea(self):
        self.__compare_analytic_for_mink_sum(True, False, 11, 5, 5, True)
        self.__compare_analytic_for_mink_sum(True, False, 12, 5, 5, True)
        self.__compare_analytic_for_mink_sum(True, False, 13, 5, 5, True)
        self.__compare_analytic_for_mink_sum(True, True, 10, 100, 100, True)

    def test_minksum_ia(self):
        self.__compare_analytic_for_mink_sum(False, False, 11, 5, 5, True)
        self.__compare_analytic_for_mink_sum(False, False, 12, 5, 5, True)
        self.__compare_analytic_for_mink_sum(False, False, 13, 5, 5, True)
        self.__compare_analytic_for_mink_sum(False, True, 10, 100, 100, True)

<<<<<<< HEAD
    def __compare_analytic_for_mink_sum(self, is_ea: bool, is_high_dim: bool, ind_typical_example: int, n_dirs: int,
                                        n_good_dirs: int, exp_result: bool):
        _compare_analytic_for_mink_sum(self, is_ea, is_high_dim, ind_typical_example, n_dirs, n_good_dirs, exp_result)

def __create_typical_high_dim_ell(ell_factory_obj, flag: int):
    if flag == 10:
        vec_1 = np.array(np.random.rand(100), dtype=np.float64)
        mat_1 = np.array(np.diag(10 * np.random.rand(100) + 0.3), dtype=np.float64)
        ell_1 = ell_factory_obj.ellipsoid(vec_1, mat_1)
        vec_2 = np.array(np.random.rand(100), dtype=np.float64)
        mat_2 = np.array(np.diag(10 * np.random.rand(100) + 0.3), dtype=np.float64)
        ell_2 = ell_factory_obj.ellipsoid(vec_2, mat_2)
        vec_3 = np.array(np.random.rand(100), dtype=np.float64)
        mat_3 = np.array(np.diag(10 * np.random.rand(100) + 0.3), dtype=np.float64)
        ell_3 = ell_factory_obj.ellipsoid(vec_3, mat_3)
        return vec_1, mat_1, vec_2, mat_2, vec_3, mat_3, np.array([ell_1, ell_2, ell_3])


def __create_typical_ell(ell_factory_obj, flag: int):
    if flag == 11:
        vec_1 = np.array([3, 61, 2, 34, 3], dtype=np.float64)
        mat_1 = np.array(5 * np.eye(5), dtype=np.float64)
        ell_1 = ell_factory_obj.ellipsoid(vec_1, mat_1)
        vec_2 = 0
        mat_2 = 0
        vec_3 = 0
        mat_3 = 0
        ell_vec = np.array([ell_1])
        return vec_1, mat_1, vec_2, mat_2, vec_3, mat_3, ell_vec
    if flag == 12:
        vec_1 = np.array([3, 61, 2, 34, 3], dtype=np.float64)
        mat_1 = np.array(5 * np.eye(5), dtype=np.float64)
        ell_1 = ell_factory_obj.ellipsoid(vec_1, mat_1)
        vec_2 = np.array([31, 34, 51, 42, 3], dtype=np.float64)
        mat_2 = np.array(np.diag([13, 3, 22, 2, 24]), dtype=np.float64)
        ell_2 = ell_factory_obj.ellipsoid(vec_2, mat_2)
        vec_3 = np.array([3, 8, 23, 12, 6], dtype=np.float64)
        mat_3 = np.array(np.diag([7, 6, 6, 8, 2]), dtype=np.float64)
        ell_3 = ell_factory_obj.ellipsoid(vec_3, mat_3)
        ell_vec = np.array([ell_1, ell_2, ell_3])
        return vec_1, mat_1, vec_2, mat_2, vec_3, mat_3, ell_vec
    if flag == 13:
        vec_1 = np.array([32, 0, 8, 1, 23], dtype=np.float64)
        mat_1 = np.array(np.diag([3, 5, 6, 5, 2]), dtype=np.float64)
        ell_1 = ell_factory_obj.ellipsoid(vec_1, mat_1)
        vec_2 = np.array([7, 3, 5, 42, 3], dtype=np.float64)
        mat_2 = np.array(np.diag([32, 34, 23, 12, 21]), dtype=np.float64)
        ell_2 = ell_factory_obj.ellipsoid(vec_2, mat_2)
        vec_3 = np.array([32, 81, 36, -25, -62], dtype=np.float64)
        mat_3 = np.array(np.diag([4, 12, 1, 1, 75]), dtype=np.float64)
        ell_3 = ell_factory_obj.ellipsoid(vec_3, mat_3)
        ell_vec = np.array([ell_1, ell_2, ell_3])
        return vec_1, mat_1, vec_2, mat_2, vec_3, mat_3, ell_vec

def __calc_exp_mink_sum(ell_factory_obj, is_ext_apx: bool, n_dirs: int, a_mat: np.ndarray,
                        e0_vec: np.ndarray, e0_mat: np.ndarray, e1_vec: np.ndarray, e1_mat: np.ndarray,
                        e2_vec: np.ndarray, e2_mat: np.ndarray) -> np.ndarray:
        from scipy.linalg import sqrtm
        analytic_res_vec = e0_vec + e1_vec + e2_vec
        analytic_res_ell_vec = np.empty(n_dirs, dtype=object)
        for i_dir in range(n_dirs):
            l_vec = a_mat[:, i_dir]
            if is_ext_apx:
                a0 = np.sqrt(l_vec @ (e0_mat @ l_vec))
                a1 = np.sqrt(l_vec @ (e1_mat @ l_vec))
                a2 = np.sqrt(l_vec @ (e2_mat @ l_vec))
                analytic_res_mat = (a0 + a1 + a2) * (e0_mat / a0 + e1_mat / a1 + e2_mat / a2)
            else:
                supp1_mat, _ = sqrtm(e0_mat, disp=False)
                supp2_mat, _ = sqrtm(e1_mat, disp=False)
                supp3_mat, _ = sqrtm(e2_mat, disp=False)
                supp1_l_vec = supp1_mat @ l_vec
                supp2_l_vec = supp2_mat @ l_vec
                supp3_l_vec = supp3_mat @ l_vec
                unitary_u1_mat, _, unitary_v1_mat = np.linalg.svd(np.expand_dims(supp1_l_vec, 1), full_matrices=True)
                unitary_u2_mat, _, unitary_v2_mat = np.linalg.svd(np.expand_dims(supp2_l_vec, 1), full_matrices=True)
                unitary_u3_mat, _, unitary_v3_mat = np.linalg.svd(np.expand_dims(supp3_l_vec, 1), full_matrices=True)
                if unitary_v1_mat.size == 1:
                    unitary_v1_mat = unitary_v1_mat * np.eye(unitary_u1_mat.shape[1])
                if unitary_v2_mat.size == 1:
                    unitary_v2_mat = unitary_v2_mat * np.eye(unitary_v1_mat.shape[1])
                if unitary_v3_mat.size == 1:
                    unitary_v3_mat = unitary_v3_mat * np.eye(unitary_v1_mat.shape[1])
                s2_mat = unitary_u1_mat @ unitary_v1_mat @ unitary_v2_mat.T @ unitary_u2_mat.T
                s2_mat = np.real(s2_mat)
                s3_mat = unitary_u1_mat @ unitary_v1_mat @ unitary_v3_mat.T @ unitary_u3_mat.T
                s3_mat = np.real(s3_mat)
                q_star_mat = supp1_mat + s2_mat @ supp2_mat + s3_mat @ supp3_mat
                analytic_res_mat = q_star_mat.T @ q_star_mat
            analytic_res_ell_vec[i_dir] = ell_factory_obj.ellipsoid(analytic_res_vec, analytic_res_mat)
        return analytic_res_ell_vec

def _compare_analytic_for_mink_sum(ell_factory_obj, is_ea: bool, is_high_dim: bool,
                                   ind_typical_example: int, n_dirs: int, n_good_dirs: int, exp_result: bool):
        if is_high_dim:
            [e0_vec, e0_mat, e1_vec, e1_mat, e2_vec, e2_mat, a_ell_vec] = \
                __create_typical_high_dim_ell(ell_factory_obj,
                                              ind_typical_example)
        else:
            [e0_vec, e0_mat, e1_vec, e1_mat, e2_vec, e2_mat, a_ell_vec] = \
                __create_typical_ell(ell_factory_obj,
                                     ind_typical_example)
        a_mat = np.array(np.eye(n_dirs), dtype=np.float64)
        if is_ea:
            test_res = a_ell_vec[0].minksum_ea(a_ell_vec, a_mat)
        else:
            test_res = a_ell_vec[0].minksum_ia(a_ell_vec, a_mat)
        if ~is_high_dim and (ind_typical_example == 11):
            test0_ell = ell_factory_obj.ellipsoid(e0_vec, e0_mat)
            analytic_res_ell_vec = test0_ell.rep_mat([5])
            is_eq_vec, report_str = analytic_res_ell_vec[0].is_equal(analytic_res_ell_vec, test_res)
            is_eq = all(is_eq_vec)
            assert is_eq is True, report_str
        else:
            analytic_res_ell_vec = __calc_exp_mink_sum(ell_factory_obj, is_ea, n_good_dirs, a_mat, e0_vec, e0_mat,
                                                       e1_vec, e1_mat, e2_vec, e2_mat)
            is_eq_vec, report_str = analytic_res_ell_vec[0].is_equal(analytic_res_ell_vec, test_res)
            is_eq = all(is_eq_vec)
            assert exp_result == is_eq, report_str