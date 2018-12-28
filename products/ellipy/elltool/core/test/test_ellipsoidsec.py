from ellipy.elltool.core.core import *
from ellipy.elltool.core.hyperplane.Hyperplane import *


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
        self.compare_analytic_for_mink_sum(True, False, 11, 5, 5, True)
        self.compare_analytic_for_mink_sum(True, False, 12, 5, 5, True)
        self.compare_analytic_for_mink_sum(True, False, 13, 5, 5, True)
        self.compare_analytic_for_mink_sum(True, True, 10, 100, 100, True)

    def test_minksum_ia(self):
        self.compare_analytic_for_mink_sum(False, False, 11, 5, 5, True)
        self.compare_analytic_for_mink_sum(False, False, 12, 5, 5, True)
        self.compare_analytic_for_mink_sum(False, False, 13, 5, 5, True)
        self.compare_analytic_for_mink_sum(False, True, 10, 100, 100, True)

    def compare_analytic_for_mink_sum(self, is_ea: bool, is_high_dim: bool, ind_typical_example: int, n_dirs: int,
                                      n_good_dirs: int, exp_result: bool):
        compare_analytic_for_mink_sum(self, is_ea, is_high_dim, ind_typical_example, n_dirs, n_good_dirs, exp_result)


def create_typical_high_dim_ell(ell_factory_obj: TestEllipsoidSecTestCase, ind_example: int):
    if ind_example == 10:
        vec_1 = np.random.rand(100)
        mat_1 = np.diag(10 * np.random.rand(100) + 0.3)
        ell_1 = ell_factory_obj.ellipsoid(vec_1, mat_1)
        vec_2 = np.random.rand(100)
        mat_2 = np.diag(10 * np.random.rand(100) + 0.3)
        ell_2 = ell_factory_obj.ellipsoid(vec_2, mat_2)
        vec_3 = np.random.rand(100)
        mat_3 = np.diag(10 * np.random.rand(100) + 0.3)
        ell_3 = ell_factory_obj.ellipsoid(vec_3, mat_3)
        return vec_1, mat_1, vec_2, mat_2, vec_3, mat_3, np.array([ell_1, ell_2, ell_3])


def create_typical_ell(ell_factory_obj: TestEllipsoidSecTestCase, ind_example: int):
    if ind_example == 11:
        vec_1 = np.array([3, 61, 2, 34, 3])
        mat_1 = 5 * np.eye(5)
        ell_1 = ell_factory_obj.ellipsoid(vec_1, mat_1)
        vec_2 = 0
        mat_2 = 0
        vec_3 = 0
        mat_3 = 0
        ell_vec = np.array([ell_1])
        return vec_1, mat_1, vec_2, mat_2, vec_3, mat_3, ell_vec
    if ind_example == 12:
        vec_1 = np.array([3, 61, 2, 34, 3])
        mat_1 = 5 * np.eye(5)
        ell_1 = ell_factory_obj.ellipsoid(vec_1, mat_1)
        vec_2 = np.array([31, 34, 51, 42, 3])
        mat_2 = np.diag([13, 3, 22, 2, 24])
        ell_2 = ell_factory_obj.ellipsoid(vec_2, mat_2)
        vec_3 = np.array([3, 8, 23, 12, 6])
        mat_3 = np.diag([7, 6, 6, 8, 2])
        ell_3 = ell_factory_obj.ellipsoid(vec_3, mat_3)
        ell_vec = np.array([ell_1, ell_2, ell_3])
        return vec_1, mat_1, vec_2, mat_2, vec_3, mat_3, ell_vec
    if ind_example == 13:
        vec_1 = np.array([32, 0, 8, 1, 23])
        mat_1 = np.diag ([3, 5, 6, 5, 2])
        ell_1 = ell_factory_obj.ellipsoid(vec_1, mat_1)
        vec_2 = np.array([7, 3, 5, 42, 3])
        mat_2 = np.diag ([32, 34, 23, 12, 21])
        ell_2 = ell_factory_obj.ellipsoid(vec_2, mat_2)
        vec_3 = np.array([32, 81, 36, -25, -62])
        mat_3 = np.diag ([4, 12, 1, 1, 75])
        ell_3 = ell_factory_obj.ellipsoid(vec_3, mat_3)
        ell_vec = np.array([ell_1, ell_2, ell_3])
        return vec_1, mat_1, vec_2, mat_2, vec_3, mat_3, ell_vec


def calc_exp_mink_sum(ell_factory_obj: TestEllipsoidSecTestCase, is_ext_apx: bool, n_dirs: int, a_mat: np.ndarray,
                      e0_vec: np.ndarray, e0_mat: np.ndarray, e1_vec: np.ndarray, e1_mat: np.ndarray,
                      e2_vec: np.ndarray, e2_mat: np.ndarray) -> np.ndarray:
        from scipy.linalg import sqrtm
        analytic_res_vec = e0_vec + e1_vec + e2_vec
        analytic_res_ell_vec = np.empty(n_dirs, dtype = object)
        for i_dir in range(n_dirs):
            l_vec = a_mat[:, i_dir]
            if is_ext_apx:
                a0 = np.sqrt(l_vec @ (e0_mat @ l_vec))
                a1 = np.sqrt(l_vec @ (e1_mat @ l_vec))
                a2 = np.sqrt(l_vec @ (e2_mat @ l_vec))
                analytic_res_mat = (a0 + a1 + a2) * (e0_mat / a0 + e1_mat / a1 + e2_mat / a2)
            else:
                supp1_mat = sqrtm(e0_mat)
                supp2_mat = sqrtm(e1_mat)
                supp3_mat = sqrtm(e2_mat)
                supp1_l_vec = supp1_mat @ l_vec
                supp2_l_vec = supp2_mat @ l_vec
                supp3_l_vec = supp3_mat @ l_vec
                unitary_u1, tmp, unitary_v1 = np.linalg.svd(supp1_l_vec)
                unitary_u2, tmp, unitary_v2 = np.linalg.svd(supp2_l_vec)
                unitary_u3, tmp, unitary_v3 = np.linalg.svd(supp3_l_vec)
                s2_mat = unitary_u1 @ unitary_v1 @ unitary_v2.T @ unitary_u2.T
                s2_mat = np.real(s2_mat)
                s3_mat = unitary_u1 @ unitary_v1 @ unitary_v3.T @ unitary_u3.T
                s3_mat = np.real(s3_mat)
                q_star_mat = supp1_mat + s2_mat @ supp2_mat + s3_mat @ supp3_mat
                analytic_res_mat = q_star_mat.T @ q_star_mat
            analytic_res_ell_vec[i_dir] =  ell_factory_obj.ellipsoid(analytic_res_vec, analytic_res_mat)
        return analytic_res_ell_vec


def compare_analytic_for_mink_sum(ell_factory_obj: TestEllipsoidSecTestCase, is_ea: bool, is_high_dim: bool,
                                  ind_typical_example: int, n_dirs: int, n_good_dirs: int, exp_result: bool):
        if is_high_dim:
            [e0_vec, e0_mat, e1_vec, e1_mat, e2_vec, e2_mat, a_ell_vec] = create_typical_high_dim_ell(ell_factory_obj,
                                                                                                     ind_typical_example)
        else:
            [e0_vec, e0_mat, e1_vec, e1_mat, e2_vec, e2_mat, a_ell_vec] = create_typical_ell(ell_factory_obj,
                                                                                            ind_typical_example)
        a_mat = np.eye(n_dirs)
        if is_ea:
            test_res = a_ell_vec[0].minksum_ea(a_ell_vec, a_mat)
        else:
            test_res = a_ell_vec[0].minksum_ia(a_ell_vec, a_mat)
        if ~is_high_dim and (ind_typical_example == 11):
            test0_ell = ell_factory_obj.ellipsoid(e0_vec, e0_mat)
            analytic_res_ell_vec = np.array([test0_ell, test0_ell.get_copy([test0_ell]).flat[0],
                                    test0_ell.get_copy([test0_ell]).flat[0],
                                    test0_ell.get_copy([test0_ell]).flat[0],
                                    test0_ell.get_copy([test0_ell]).flat[0]])
            is_eq_vec, report_str = analytic_res_ell_vec[0].is_equal(analytic_res_ell_vec, test_res)
            is_eq= all(is_eq_vec)
            assert is_eq is True #how to disp report_str
        else:
            analytic_res_ell_vec = calc_exp_mink_sum(ell_factory_obj,  is_ea, n_good_dirs, a_mat, e0_vec, e0_mat,
                                                     e1_vec, e1_mat, e2_vec, e2_mat)
            is_eq_vec, report_str = analytic_res_ell_vec[0].is_equal(analytic_res_ell_vec, test_res)
            is_eq= all(is_eq_vec)
            assert exp_result == is_eq

