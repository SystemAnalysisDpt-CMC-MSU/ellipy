from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.gen.common.common import throw_error
from ellipy.gen.common.common import abs_rel_compare
from ellipy.gras.geom.tri.tri import is_tri_equal
from ellipy.gras.gen.gen import sort_rows_tol
from typing import Tuple
import os
import scipy.io
import pytest
import copy


class TestEllipsoidBasicSecondTC:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    def test_uminus(self):
        test_1_ell = self.ellipsoid(np.array([0, 0]),
                                    np.array([
                                        [1, 0],
                                        [0, 1]
                                    ]))
        test_2_ell = self.ellipsoid(np.array([1, 0]),
                                    np.array([
                                        [1, 0],
                                        [0, 1]
                                    ]))
        test_3_ell = self.ellipsoid(np.array([1, 0]),
                                    np.array([
                                        [2, 0],
                                        [0, 1]
                                    ]))
        test_4_ell = self.ellipsoid(np.array([0, 0]),
                                    np.array([
                                        [0, 0],
                                        [0, 0]
                                    ]))
        test_5_ell = self.ellipsoid(np.array([0, 0, 0]),
                                    np.array([
                                        [0, 0, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]
                                    ]))
        test_6_ell = self.ellipsoid()
        test_7_ell = self.ellipsoid(np.array([2, 1]),
                                    np.array([
                                        [3, 1],
                                        [1, 1]
                                    ]))
        test_8_ell = self.ellipsoid(np.array([1, 1]),
                                    np.array([
                                        [1, 0],
                                        [0, 1]
                                    ]))
        check_center_vec_list = [np.array([-1, 0])]
        self.__operation_check_eq_func(test_2_ell, check_center_vec_list, 'uminus')
        #
        test_ell_vec = [test_1_ell, test_2_ell, test_3_ell]
        check_center_vec_list = [np.array([0, 0]),
                                 np.array([-1, 0]),
                                 np.array([-1, 0])]
        self.__operation_check_eq_func(test_ell_vec, check_center_vec_list, 'uminus')
        #
        test_ell_mat = [[test_1_ell, test_2_ell], [test_3_ell, test_4_ell]]
        check_center_vec_list = [[np.array([0, 0]),
                                  np.array([-1, 0])],
                                 [np.array([-1, 0]),
                                  np.array([0, 0])]]
        self.__operation_check_eq_func(test_ell_mat, check_center_vec_list, 'uminus')
        #
        test_ell_vec = [test_1_ell, test_2_ell, test_3_ell, test_4_ell,
                        test_5_ell, test_6_ell, test_7_ell, test_8_ell]
        test_ell_arr = np.reshape(test_ell_vec, newshape=(2, 2, 2))
        check_center_vec_list = np.reshape([np.array([0, 0]),
                                            np.array([-1, 0]),
                                            np.array([-1, 0]),
                                            np.array([0, 0]),
                                            np.array([0, 0, 0]),
                                            np.array([]),
                                            np.array([-2, -1]),
                                            np.array([-1, -1])], newshape=(2, 2, 2)).tolist()
        self.__operation_check_eq_func(test_ell_arr, check_center_vec_list, 'uminus')
        #
        test_ell_center_vec = np.zeros(shape=(100, 1))
        test_ell_center_vec[49] = 1
        test_ell_mat = np.eye(100)
        test_ell = self.ellipsoid(test_ell_center_vec, test_ell_mat)
        test_res_vec = np.zeros(shape=(100, 1))
        test_res_vec[49] = -1
        check_center_vec_list = [test_res_vec]
        self.__operation_check_eq_func(test_ell, check_center_vec_list, 'uminus')
        #
        self.__empty_test('uminus', [0, 0, 2, 0])

    def __operation_check_eq_func(self, test_ell_arr, comp_list, operation, argument=None):
        comp_list = np.array(comp_list)
        test_ell_arr = np.array(test_ell_arr)
        __OBJ_MODIFICATING_METHODS_LIST = ['inv', 'move_2_origin', 'shape']
        is_obj_modif_method = np.isin(operation, __OBJ_MODIFICATING_METHODS_LIST)
        test_copy_ell_arr = []
        if ~is_obj_modif_method:
            test_copy_ell_arr = copy.deepcopy(test_ell_arr)
        if test_ell_arr.size > 0:
            ell_class = test_ell_arr.flat[0].__class__
        else:
            ell_class = self.ellipsoid().__class__
        if argument is None:
            test_ell_res_arr = getattr(ell_class, operation)(test_ell_arr.flatten())
        else:
            test_ell_res_arr = getattr(ell_class, operation)(test_ell_arr.flatten(), argument)
        self.__check_res(test_ell_res_arr, comp_list, operation)
        if is_obj_modif_method:
            # test for methods which modify the input array
            self.__check_res(test_ell_arr, comp_list, operation)
        else:
            # test for absence of input array's modification
            is_eq_arr, report_str = ell_class.is_equal(test_copy_ell_arr, test_ell_arr)
            is_not_modif = np.all(is_eq_arr)
            assert is_not_modif, report_str

    @staticmethod
    def __check_res(test_ell_res_arr, comp_list, operation):
        __VEC_COMP_METHODS_LIST = ['uminus', 'plus', 'minus', 'move_2_origin', 'get_move_2_origin']
        __MAT_COMP_METHODS_LIST = ['inv', 'shape', 'get_inv', 'get_shape']
        #
        test_ell_res_centers_vec_list = np.array([[elem.get_center_vec() for elem in test_ell_res_arr.flatten()]])
        test_ell_res_shape_mat_list = np.array([[elem.get_shape_mat() for elem in test_ell_res_arr.flatten()]])
        if np.isin(operation, __VEC_COMP_METHODS_LIST):
            eq_arr = [np.array_equal(x, y) for (x, y) in
                      zip(test_ell_res_centers_vec_list.flatten(), comp_list.flatten())]
        elif np.isin(operation, __MAT_COMP_METHODS_LIST):
            eq_arr = [np.array_equal(x, y) for (x, y) in
                      zip(test_ell_res_shape_mat_list.flatten(), comp_list.flatten())]
        else:
            eq_arr = []
            expr = ' '.join(__VEC_COMP_METHODS_LIST + __MAT_COMP_METHODS_LIST)
            throw_error('wrongInput:badMethodName', 'Allowed method names: {}. Input name: {}'.format(expr, operation))
        test_is_right = np.all(np.equal(eq_arr[:], 1))
        assert test_is_right

    def __empty_test(self, method_name, size_vec, argument=None):
        test_ell_arr = np.empty(shape=size_vec, dtype=self.ellipsoid().__class__)
        check_center_vec_list = np.tile([], size_vec)
        if argument is None:
            self.__operation_check_eq_func(test_ell_arr, check_center_vec_list, method_name)
        else:
            self.__operation_check_eq_func(test_ell_arr, check_center_vec_list, method_name, argument)

    def test_projection(self):
        project_mat = np.array([[1, 0], [0, 1], [0, 0]])
        cent_vec = np.array([[-2], [-1], [4]])
        shape_mat = np.array([[4, -1, 0], [-1, 1, 0], [0, 0, 9]])
        self.__aux_test_projection('projection', cent_vec, shape_mat, project_mat)
        #
        project_mat = np.array([[1, 0], [0, 0], [0, 1]])
        cent_vec = np.array([[2], [4], [3]])
        shape_mat = np.array([[3, 1, 1], [1, 4, 1], [1, 1, 8]])
        dim_vec = np.array([[2, 2, 3, 4]])
        self.__aux_test_projection('projection', cent_vec, shape_mat, project_mat, dim_vec)
        #
        dim_vec = np.array([[0, 0, 2, 0]])
        self.__aux_test_projection('projection', cent_vec, shape_mat, project_mat, dim_vec)

    def test_get_projection(self):
        project_mat = np.array([[1, 0], [0, 0], [0, 1]])
        cent_vec = np.array([[2], [4], [3]])
        shape_mat = np.array([[3, 1, 1], [1, 4, 1], [1, 1, 8]])
        dim_vec = np.array([[2, 2, 3, 4]])
        self.__aux_test_projection('get_projection', cent_vec, shape_mat, project_mat, dim_vec)
        #
        dim_vec = np.array([[0, 0, 2, 0]])
        self.__aux_test_projection('get_projection', cent_vec, shape_mat, project_mat, dim_vec)

    def __aux_test_projection(self, *args, **kwargs):
        self.aux_test_projection(self, *args, **kwargs)

    @staticmethod
    def aux_test_projection(ell_fact_obj, method_name: str,
                            cent_vec: np.ndarray, shape_mat: np.ndarray,
                            pr_mat: np.ndarray, dim_vec: np.ndarray = None):

        def is_equ_internal(ell_obj_1_vec: Union[Iterable, np.ndarray],
                            ell_obj_2_vec: Union[Iterable, np.ndarray]) -> bool:
            ell_obj_1_vec = np.array(ell_obj_1_vec)
            ell_obj_2_vec = np.array(ell_obj_2_vec)
            if ell_obj_1_vec.size == 0 or ell_obj_2_vec.size == 0:
                is_ok = np.array_equal(ell_obj_1_vec.shape, ell_obj_2_vec.shape)
            else:
                cent_vec_1_vec, shape_mat_1_vec = zip(*map(lambda ell: ell.double(),
                                                           ell_obj_1_vec.flatten()))
                cent_vec_2_vec, shape_mat_2_vec = zip(*map(lambda ell: ell.double(),
                                                           ell_obj_2_vec.flatten()))
                is_ok = np.array_equal(cent_vec_1_vec, cent_vec_2_vec) and \
                    np.array_equal(shape_mat_1_vec, shape_mat_2_vec) and \
                    np.array_equal(ell_obj_1_vec.shape, ell_obj_2_vec.shape)
            return is_ok

        is_inp_obj_modify = None
        ell_copy_obj = None
        ell_copy_arr = None
        inp_obj_modify_list = ['projection']
        inp_obj_not_modify_list = ['get_projection']
        pr_cent_vec = pr_mat.T @ cent_vec
        pr_shape_mat = pr_mat.T @ shape_mat @ pr_mat
        ell_obj = ell_fact_obj.ellipsoid(cent_vec, shape_mat)
        comp_ell_obj = ell_fact_obj.ellipsoid(pr_cent_vec, pr_shape_mat)
        if method_name in inp_obj_modify_list:
            is_inp_obj_modify = True
        elif method_name in inp_obj_not_modify_list:
            ell_copy_obj = ell_obj.get_copy(ell_obj).flat[0]
            is_inp_obj_modify = False
        else:
            throw_error('wrongInput:bad_method_name',
                        'Allowed method names: ' + inp_obj_modify_list[0] +
                        ', ' + inp_obj_not_modify_list[0] + '. Input name: ' + method_name)
        if dim_vec is None:
            pr_ell_obj = getattr(ell_obj, method_name)(ell_obj, pr_mat)
            test_is_right_1 = is_equ_internal(comp_ell_obj, pr_ell_obj)
            if is_inp_obj_modify:
                # additional test for modification of input object
                test_is_right_2, _ = comp_ell_obj.is_equal(comp_ell_obj, ell_obj)
            else:
                # additional test for absence of input object's modification
                test_is_right_2, _ = ell_copy_obj.is_equal(ell_copy_obj, ell_obj)
        else:
            ell_arr = ell_obj.rep_mat(dim_vec)
            if not is_inp_obj_modify:
                ell_copy_arr = ell_copy_obj.rep_mat(dim_vec)
            pr_ell_arr = getattr(ell_obj, method_name)(ell_arr, pr_mat)
            comp_ell_arr = comp_ell_obj.rep_mat(dim_vec)
            test_is_right_1 = is_equ_internal(comp_ell_arr, pr_ell_arr)
            if is_inp_obj_modify:
                # additional test for modification of input array
                test_arr_2, _ = comp_ell_obj.is_equal(comp_ell_arr, ell_arr)
                test_is_right_2 = np.all(test_arr_2)
            else:
                # additional test for absence of input array's modification
                test_arr_2, _ = ell_copy_obj.is_equal(ell_copy_arr, ell_arr)
                test_is_right_2 = np.all(test_arr_2)
        assert test_is_right_1
        assert test_is_right_2

    def test_rho(self):
        def check_rho_res(sup_arr_check: np.ndarray, bp_arr_check: np.ndarray):
            is_rho_ok = np.all(sup_arr_check[:] == 5)
            is_bp_ok = np.all(bp_arr_check[0, :] == 5) and np.all(bp_arr_check[1, :] == 0)
            assert is_rho_ok and is_bp_ok

        def check_rho_size(sup_arr_check: np.ndarray, bp_arr_check: np.ndarray,
                           dir_arr_check: np.ndarray, arr_size_vec_check: np.ndarray):
            is_rho_ok = np.all(sup_arr_check.shape == arr_size_vec_check)
            is_bp_ok = bp_arr_check.shape == dir_arr_check.shape
            assert is_rho_ok and is_bp_ok

        #
        dir_mat = np.array([[1, 1], [0, 0]])
        ell_obj_mat = np.diag(np.array([9, 25]))
        ell_obj_cen_vec = np.array([[2], [0]])
        ell_obj = self.ellipsoid(ell_obj_cen_vec, ell_obj_mat)
        ell_obj_arr = np.array([ell_obj])
        ell_vec = np.array([[ell_obj, ell_obj, ell_obj]])
        #
        # Check one ell - one dirs
        sup_val, bp_vec = ell_obj.rho(ell_obj_arr, dir_mat[:, 0].reshape(-1, 1))
        check_rho_res(sup_val, bp_vec)
        check_rho_size(sup_val, bp_vec, np.ones((2, 1)), np.array([[1, 1]]))
        #
        # Check one ell - multiple dirs
        sup_arr, bp_mat = ell_obj.rho(ell_obj_arr, dir_mat)
        check_rho_res(sup_arr, bp_mat)
        check_rho_size(sup_arr, bp_mat, dir_mat, np.array([[1, 2]]))
        #
        # Check multiple ell - one dir
        sup_arr, bp_mat = ell_obj.rho(ell_vec, dir_mat[:, 0].reshape(-1, 1))
        check_rho_res(sup_arr, bp_mat)
        check_rho_size(sup_arr, bp_mat, np.ones((2, 3)), np.array([[1, 3]]))
        #
        # Check multiple ell - multiple dirs
        arr_size_vec = np.array([2, 3, 4])
        dir_arr = np.zeros((2,) + tuple(arr_size_vec))
        dir_arr[0, :] = 1
        test_ell = self.ellipsoid(ell_obj_cen_vec, ell_obj_mat)
        ell_arr = test_ell.rep_mat(arr_size_vec)
        sup_arr, bp_arr = ell_obj.rho(ell_arr, dir_arr)
        check_rho_res(sup_arr, bp_arr)
        check_rho_size(sup_arr, bp_arr, dir_arr, arr_size_vec)
        #
        # Check array ell - one dir
        sup_arr, bp_arr = ell_obj.rho(ell_arr, dir_mat[:, 0].reshape(-1, 1))
        check_rho_res(sup_arr, bp_arr)
        check_rho_size(sup_arr, bp_arr, dir_arr, arr_size_vec)
        #
        # Check one ell - array dir
        sup_arr, bp_arr = ell_obj.rho(ell_obj_arr, dir_arr)
        check_rho_res(sup_arr, bp_arr)
        check_rho_size(sup_arr, bp_arr, dir_arr, arr_size_vec)
        #
        # Negative tests for input
        arr2_size_vec = np.array([2, 2, 4])
        dir2_arr = np.ones((2,) + tuple(arr2_size_vec))
        test_ell = self.ellipsoid(ell_obj_cen_vec, ell_obj_mat)
        ell2_arr = test_ell.rep_mat(arr2_size_vec)
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            ell_obj.rho(ell2_arr, dir_arr)
        assert 'wrongInput:wrongSizes' in str(e.value)
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            ell_obj.rho(ell_arr, dir2_arr)
        assert 'wrongInput:wrongSizes' in str(e.value)
        ell_vec = np.array([[ell_obj, ell_obj, ell_obj]])
        dir_mat = np.eye(2)
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            ell_obj.rho(ell_vec, dir_mat)
        assert 'wrongInput:wrongSizes' in str(e.value)
        ell_vec = np.array([[ell_obj, ell_obj, ell_obj]]).T
        dir_mat = np.eye(2)
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            ell_obj.rho(ell_vec, dir_mat)
        assert 'wrongInput:wrongSizes' in str(e.value)
        ell_empty_arr = np.empty((0, 0, 2, 0), dtype=np.object)
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            ell_obj.rho(ell_empty_arr, dir_mat)
        assert 'wrongInput:wrongSizes' in str(e.value)

    def test_get_boundary(self):
        test_ell_vec, test_num_points_vec = self.__get_ell_params(self, 1)
        data_size = test_ell_vec.size
        bp_arr = np.zeros(data_size, dtype=np.ndarray)
        f_arr = np.zeros(data_size, dtype=np.ndarray)
        bp_right_arr = np.zeros(data_size, dtype=np.ndarray)
        f_right_arr = np.zeros(data_size, dtype=np.ndarray)
        bp_right_data_list = [np.array([[1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2], [-1, 0],
                                        [-0.5, -np.sqrt(3) / 2], [0.5, -np.sqrt(3) / 2]], dtype=np.float64),
                              np.array([[2, 0], [1.5, np.sqrt(3) / 2], [0.5, np.sqrt(3) / 2], [0, 0],
                                        [0.5, -np.sqrt(3) / 2], [1.5, -np.sqrt(3) / 2]], dtype=np.float64),
                              np.zeros((6, 2), dtype=np.float64),
                              np.array([[4, 1], [3, np.sqrt(3) + 1], [1, np.sqrt(3) + 1], [0, 1],
                                        [1, -np.sqrt(3) + 1], [3, -np.sqrt(3) + 1]], dtype=np.float64)]
        for i in range(data_size):
            bp_arr[i], f_arr[i] = test_ell_vec[i].get_boundary(test_num_points_vec[i], True)
            bp_right_arr[i] = bp_right_data_list[i]
            f_right_arr[i] = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]], dtype=np.float64)
        is_ok = self.__compare_cells(bp_arr, f_arr, bp_right_arr, f_right_arr)
        assert is_ok

    def test_get_boundary_by_factor(self):
        test_ell_vec, test_num_points_vec = self.__get_ell_params(self, 1)
        data_size = test_ell_vec.size
        bp_arr = np.zeros(data_size, dtype=np.ndarray)
        f_arr = np.zeros(data_size, dtype=np.ndarray)
        bp_right_arr = np.zeros(data_size, dtype=np.ndarray)
        f_right_arr = np.zeros(data_size, dtype=np.ndarray)
        test_num_right_points_vec = np.zeros(data_size, dtype=np.int)

        for i in range(data_size):
            test_num_right_points_vec[i] = test_ell_vec[i].get_n_plot_2d_points(test_ell_vec[i]) * test_num_points_vec[
                i]
            bp_arr[i], f_arr[i] = test_ell_vec[i].get_boundary_by_factor(test_num_points_vec[i], True)
            bp_right_arr[i], f_right_arr[i] = test_ell_vec[i].get_boundary(test_num_right_points_vec[i], True)
        is_ok = self.__compare_cells(bp_arr, f_arr, bp_right_arr, f_right_arr)
        assert is_ok

    def test_get_rho_boundary_by_factor(self):
        def get_num_points(ell: Ellipsoid, factor_vec: Union[int, np.ndarray]) -> int:
            n_dims = ell.dimension([ell])
            if n_dims == 2:
                return int(ell._n_plot_2d_points) * factor_vec
            else:
                return int(ell._n_plot_3d_points) * factor_vec[n_dims - 2]

        test_ell_vec, test_num_points_vec = self.__get_ell_params(self, 2)
        data_size = test_ell_vec.size
        test_num_right_points_vec = np.zeros(data_size, dtype=np.int)
        for i in range(data_size):
            test_num_right_points_vec[i] = get_num_points(test_ell_vec[i], test_num_points_vec[i])
            tuple_got = test_ell_vec[i].get_rho_boundary_by_factor(test_num_points_vec[i])
            tuple_expected = test_ell_vec[i].get_rho_boundary(test_num_right_points_vec[i])
            is_ok = np.all([np.allclose(tuple_expected[j], tuple_got[j]) for j in range(len(tuple_got))])
            assert is_ok

    def test_neg_boundary(self):
        # noinspection PyChecker,PyUnboundLocalVariable
        def run_and_check_error(func: Callable, val: str):
            try:
                func()
            except Exception as e:
                split_str = str(e).split(":")
                is_ok = (split_str[-2] == val) or (split_str[-3] == val)
                assert is_ok

        # noinspection PyChecker
        def check_dim():

            def check_dim_g_b():
                ell_obj = self.ellipsoid(np.eye(4))
                ell_obj.get_boundary()

            def check_dim_g_b_b_f():
                ell_obj = self.ellipsoid(np.eye(4))
                ell_obj.get_boundary_by_factor()

            def check_dim_g_r_b():
                ell_obj = self.ellipsoid(np.eye(4))
                ell_obj.get_rho_boundary()

            def check_dim_g_r_b_b_f():
                ell_obj = self.ellipsoid(np.eye(4))
                ell_obj.get_rho_boundary_by_factor()

            errmsg = 'wrongDim'
            run_and_check_error(check_dim_g_b, errmsg)
            run_and_check_error(check_dim_g_b_b_f, errmsg)
            run_and_check_error(check_dim_g_r_b, errmsg)
            run_and_check_error(check_dim_g_r_b_b_f, errmsg)

        # noinspection PycChecker
        def check_scal():
            ell_vec = np.array([self.ellipsoid(np.array([[1], [3]]), np.eye(2)),
                                self.ellipsoid(np.array([[2], [5]]), np.array([[4, 1], [1, 1]]))])

            # noinspection PyTypeChecker,PyCallByClass
            def check_scal_g_b():
                ell_vec[0].get_boundary()

            # noinspection PyTypeChecker,PyCallByClass
            def check_scal_g_b_b_f():
                ell_vec[0].get_boundary_by_factor()

            # noinspection PyTypeChecker,PyCallByClass
            def check_scal_g_r_b():
                ell_vec[0].get_rho_boundary()

            # noinspection PyTypeChecker,PyCallByClass
            def check_scal_g_r_b_b_f():
                ell_vec[0].get_rho_boundary_by_factor()

            errmsg = 'wrongInput'
            run_and_check_error(check_scal_g_b, errmsg)
            run_and_check_error(check_scal_g_b_b_f, errmsg)
            run_and_check_error(check_scal_g_r_b, errmsg)
            run_and_check_error(check_scal_g_r_b_b_f, errmsg)

        check_dim()
        check_scal()

    def test_get_rho_boundary(self):
        __MAX_TOL__ = 1e-7
        test_ell_vec, test_num_points_vec = self.__get_ell_params(self, 2)
        data_size = test_ell_vec.size
        loaded_data = scipy.io.loadmat(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data_ellipsoidbasicsecond.mat'))
        bp_right_mat_arr = loaded_data['bpRightMatCArr']
        f_right_mat_arr = loaded_data['fRightMatCArr']
        l_grid_right_arr = loaded_data['lGridRightCMat']
        sup_right_vec = loaded_data['supRightCVec']
        ell_class = test_ell_vec.flat[0].__class__
        for i in range(data_size):
            bp_arr, f_arr, sup_vec, l_grid_arr = ell_class.get_rho_boundary(test_ell_vec[i], test_num_points_vec[i][0])
            is_lgrid_ok = np.allclose(l_grid_arr[0, :], l_grid_arr[-1, :])
            l_grid_right = l_grid_right_arr[0, i]
            bp_right = bp_right_mat_arr[0, i]
            sup_right = sup_right_vec[0, i]
            f_right = f_right_mat_arr[0, i] - 1
            v_arr = l_grid_arr[0:-1, :]
            v_right_arr = l_grid_right[0:-1, :]
            is_v_eq, _ = is_tri_equal(v_arr, f_arr, v_right_arr, f_right, __MAX_TOL__)
            l_grid_arr, ind_sort_py, _ = sort_rows_tol(l_grid_arr, __MAX_TOL__)
            l_grid_right, ind_sort_mat, _ = sort_rows_tol(l_grid_right, __MAX_TOL__)
            is_lgrid_eq = np.allclose(l_grid_arr, l_grid_right)
            is_bp_eq = np.allclose(bp_arr[ind_sort_py], bp_right[ind_sort_mat])
            is_sup_eq = np.allclose(sup_vec[ind_sort_py], sup_right[ind_sort_mat])
            assert is_lgrid_ok and is_v_eq and is_lgrid_eq and is_bp_eq and is_sup_eq

    @staticmethod
    def __get_ell_params(ell_factory_obj, flag: int) -> Tuple[np.ndarray, np.ndarray]:
        if flag == 1:
            test_1_ell = ell_factory_obj.ellipsoid(np.eye(2))
            test_2_ell = ell_factory_obj.ellipsoid(np.array([1, 0]), np.array([[1, 0], [0, 1]]))
            test_3_ell = ell_factory_obj.ellipsoid(np.array([0, 0]), np.array([[0, 0], [0, 0]]))
            test_4_ell = ell_factory_obj.ellipsoid(np.array([2, 1]), np.array([[4, 0], [0, 4]]))
            points_vec = np.array([6, 6, 6, 6], dtype=object)
        else:
            test_1_ell = ell_factory_obj.ellipsoid(np.eye(2))
            test_2_ell = ell_factory_obj.ellipsoid(np.array([1, 3]), np.array([[3, 1], [1, 1]]))
            test_3_ell = ell_factory_obj.ellipsoid(np.array([2, 1]), np.array([[4, -1], [-1, 1]]))
            test_4_ell = ell_factory_obj.ellipsoid(np.eye(3))
            points_vec = np.array([np.array([10]), np.array([20]), np.array([35]), np.array([5, 5])], dtype=object)
        ell_vec = np.array([test_1_ell, test_2_ell, test_3_ell, test_4_ell], dtype=object)
        return ell_vec, points_vec

    @staticmethod
    def __compare_cells(bp_arr: np.ndarray, f_arr: np.ndarray, bp_right_arr: np.ndarray,
                        f_right_arr: np.ndarray) -> bool:

        __ABSTOL__ = 1.0e-12
        is_equal_1 = True
        is_equal_2 = True
        if not (bp_arr.size == f_arr.size):
            throw_error('wrongInput', 'bp_arr and f_arr must be of the same size')
        for i in range(bp_arr.size):
            is_equal_1 = is_equal_1 and \
                abs_rel_compare(bp_right_arr[i], bp_arr[i], __ABSTOL__, __ABSTOL__, np.linalg.norm)[0]
            is_equal_2 = is_equal_2 and \
                abs_rel_compare(f_right_arr[i], f_arr[i], __ABSTOL__, __ABSTOL__, np.linalg.norm)[0]
        return is_equal_1 and is_equal_2
