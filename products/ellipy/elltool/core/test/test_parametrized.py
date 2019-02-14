import numpy as np
from ellipy.elltool.conf.properties.Properties import *#get_abs_tol, get_rel_tol
from ellipy.gen.common.common import throw_error

def get_def_tol():
    return np.array([Properties.get_abs_tol(), Properties.get_rel_tol()])

def check_for_is_equal(test_ell1_vec, test_ell2_vec, is_exp_res_is_equal):
    is_ok_arr = np.equal(test_ell1_vec, test_ell2_vec)
    is_ok = np.all(is_ok_arr == is_exp_res_is_equal)
    assert is_ok

class TestParametrizedTC(AEllipsoid):

    def set_up_param(self, *args):
        if np.size(args) == 2:
            self.f_create_obj = args[0]
        elif np.size(args) > 1:
            throw_error('wrongInput: too many parametres')

    def __init__(self, *args):
        if np.size(args) == 2:
            self.f_create_obj = args[0]

    def create_obj_c_vec(self, cent_c_vec, *args):
        a = cent_c_vec
        mat_c_vec = np.tile(np.eye(np.size(cent_c_vec[1])), (1, np.size(cent_c_vec)))
        if np.size(args) == 0:
            f_cr_ob_vec = self.f_create_obj(cent_c_vec, shape_mat)
        else:
            f_cr_ob_vec = self.f_create_obj(cent_c_vec, shape_mat, {'abs_tol': abs_tol, 'rel_tol': rel_tol})
        obj_c_vec = f_cr_ob_vec(cent_c_vec, mat_c_vec, args, 'UniformOutPut', 0)
        return obj_c_vec


    def test_is_equal_sym_prop(self):
    #test symmetry property
        def_tol = np.min(get_def_tol())
        tol_vec = np.array([10 * def_tol, def_tol])
        cent_c_vec = np.array([[0, tol_vec[0]], [1, 1]])
        test_obj_c_vec = self.create_obj_c_vec(cent_c_vec, tol_vec, tol_vec)
        check_for_is_equal(test_obj_c_vec[:, 1], test_obj_c_vec[:, 0],
                           np.equal(test_obj_c_vec[:, 0], test_obj_c_vec[:, 1]))

    def test_is_equal_trans_prop(self):
    #test transitive property
        def_tol = np.min(get_def_tol())
        #abs_tol_vec = rel_tol_vec = tol_vec
        tol_vec = np.array([10 * def_tol, 100 * def_tol, def_tol])
        cent_c_vec = np.array([[0, tol_vec[0], tol_vec[1], [1, 1, 1]]])
        test_obj_c_vec = self.create_obj_c_vec(cent_c_vec, tol_vec, tol_vec)
        check_for_is_equal(test_obj_c_vec[:, 0], test_obj_c_vec[:, 2],
                           np.equal(test_obj_c_vec[:, 0], test_obj_c_vec[:, 1]) and
                           np.equal(test_obj_c_vec[:, 1], test_obj_c_vec[:, 2]))

    def test_is_equal_abs_tol_rep_by_rel_tol(self):
    #test captures that abs_tol replaced by rel_tol
        def_abs_tol, def_rel_tol = get_def_tol()
        if def_abs_tol != def_rel_tol:
            if def_abs_tol < def_rel_tol:
                cent_c_vec = np.array([[0, def_rel_tol], [1, 1]])
                is_exp_res_is_equal = 0
            else:
                cent_c_vec = np.array([[0, def_abs_tol], [1, 1]])
                is_exp_res_is_equal = 1
            test_obj_c_vec = self.create_obj_c_vec(cent_c_vec)
        else:
            abs_tol_vec = np.array([def_abs_tol, def_abs_tol])
            rel_tol_vec = np.array([10 * def_abs_tol, 10 * def_abs_tol])
            cent_c_vec = np.array([[0, rel_tol_vec[0]], [1, 1]])
            test_obj_c_vec = self.create_obj_c_vec(cent_c_vec, abs_tol_vec, rel_tol_vec)
            is_exp_res_is_equal = 0
        check_for_is_equal(test_obj_c_vec[:, 0], test_obj_c_vec[:, 1], is_exp_res_is_equal)
