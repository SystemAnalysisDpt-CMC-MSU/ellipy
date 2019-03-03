import pytest
import numpy as np
from ellipy.elltool.conf.properties.Properties import *
from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.elltool.core.hyperplane.Hyperplane import *


def hyp(hyp_norm_arr, hyp_const_arr, *args, **kwargs):
    return Hyperplane(hyp_norm_arr, hyp_const_arr.flatten()[0], *args, **kwargs)


def ellipsoid(*args, **kwargs):
    return Ellipsoid(*args, **kwargs)


def get_def_tol():
    return [Properties.get_abs_tol(), Properties.get_rel_tol()]


def check_for_is_equal(test_ell1_vec, test_ell2_vec, is_exp_res_is_equal):
    is_ok_arr = np.equal(test_ell1_vec, test_ell2_vec)
    is_ok = np.all(is_ok_arr == is_exp_res_is_equal)
    assert is_ok


@pytest.fixture(
    scope="class",
    ids=['Hyperplane', 'Ellipsoid'],
    params=[hyp, ellipsoid]
)
def f_create(request):
    return request.param


class TestParametrizedTC:

    @staticmethod
    def __create_obj_list(f_create, cent_list, *args):
        shape_mat_list = [np.eye(cent_list[0].size)] * len(cent_list)
        if len(args) == 0:
            obj_list = [f_create(cent_vec, shape_mat) for (cent_vec, shape_mat) in zip(cent_list, shape_mat_list)]

        else:
            obj_list = [f_create(cent_vec, shape_mat, **{'absTol': add_args[0], 'relTol': add_args[1]}) for
                        (cent_vec, shape_mat, add_args) in zip(cent_list, shape_mat_list, args)]
        return obj_list

    def test_is_equal_sym_prop(self, f_create):
        # test symmetry property
        def_tol = min(get_def_tol())
        tol_vec = [[10 * def_tol, 10 * def_tol], [def_tol, def_tol]]
        cent_vec_list = [np.array([0, 1]), np.array([tol_vec[0][0], 1])]
        test_obj_list = self.__create_obj_list(f_create, cent_vec_list, *tol_vec)
        check_for_is_equal(test_obj_list[1], test_obj_list[0],
                           np.equal(test_obj_list[0], test_obj_list[1]))

    def test_is_equal_trans_prop(self, f_create):
        # test transitive property
        def_tol = min(get_def_tol())
        # abs_tol_vec = rel_tol_vec = tol_vec
        tol_vec = [[10 * def_tol, 10 * def_tol], [100 * def_tol, 100 * def_tol], [def_tol, def_tol]]
        cent_vec_list = [np.array([0, 1]), np.array([tol_vec[0][0], 1]), np.array([tol_vec[1][0], 1])]
        test_obj_list = self.__create_obj_list(f_create, cent_vec_list, *tol_vec)
        check_for_is_equal(test_obj_list[0], test_obj_list[2],
                           np.equal(test_obj_list[0], test_obj_list[1]) and
                           np.equal(test_obj_list[1], test_obj_list[2]))

    def test_is_equal_abs_tol_rep_by_rel_tol(self, f_create):
        # test captures that abs_tol replaced by rel_tol
        def_abs_tol, def_rel_tol = get_def_tol()
        if def_abs_tol != def_rel_tol:
            if def_abs_tol < def_rel_tol:
                cent_vec_list = [np.array([0, 1]), np.array([def_rel_tol, 1])]
                is_exp_res_is_equal = 0
            else:
                cent_vec_list = [np.array([0, 1]), np.array([def_abs_tol, 1])]
                is_exp_res_is_equal = 1
            test_obj_list = self.__create_obj_list(f_create, cent_vec_list)
        else:
            tol_vec = [[def_abs_tol, 10 * def_abs_tol], [def_abs_tol, 10 * def_abs_tol]]
            cent_vec_list = [np.array([0, 1]), np.array([tol_vec[0][0], 1])]
            test_obj_list = self.__create_obj_list(f_create, cent_vec_list, *tol_vec)
            is_exp_res_is_equal = 0
        check_for_is_equal(test_obj_list[0], test_obj_list[1], is_exp_res_is_equal)
