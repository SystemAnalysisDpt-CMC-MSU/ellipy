from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.gen.common.common import throw_error
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

        return self

    @classmethod
    def __operation_check_eq_func(cls, test_ell_arr, comp_list, operation, argument=None):
        comp_list = np.array(comp_list)
        test_ell_arr = np.array(test_ell_arr)
        __OBJ_MODIFICATING_METHODS_LIST = ['inv', 'move_2_origin', 'shape']
        is_obj_modif_method = np.isin(operation, __OBJ_MODIFICATING_METHODS_LIST)
        test_copy_ell_arr = []
        if ~is_obj_modif_method:
            test_copy_ell_arr = copy.deepcopy(test_ell_arr)
        if argument is None:
            try:
                test_ell_res_arr = np.array([eval('elem.{1}(elem)'.format(str(i), operation))
                                             for (i, elem) in enumerate(test_ell_arr.flatten())])
            except TypeError:
                test_ell_res_arr = eval('elem.{}(test_ell_arr)'.format(operation))
        else:
            try:
                test_ell_res_arr = np.array([eval('elem.{1}(elem, {2})'.format(str(i), operation, argument))
                                             for (i, elem) in enumerate(test_ell_arr.flatten())])
            except TypeError:
                test_ell_res_arr = eval('elem.{}(test_ell_arr, {})'.format(operation, argument))
        cls.__check_res(test_ell_res_arr, comp_list, operation)
        if is_obj_modif_method:
            # test for methods which modify the input array
            cls.__check_res(test_ell_arr, comp_list, operation)
        else:
            # test for absence of input array's modification
            is_not_modif = np.all(np.equal(test_copy_ell_arr, test_ell_arr))
            assert 1 == is_not_modif

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
        assert 1 == test_is_right

    @classmethod
    def __empty_test(cls, method_name, size_vec, argument=None):
        test_ell_arr = np.empty(shape=size_vec, dtype=Ellipsoid)
        check_center_vec_list = np.tile([], size_vec)
        if argument is None:
            cls.__operation_check_eq_func(test_ell_arr, check_center_vec_list, method_name)
        else:
            cls.__operation_check_eq_func(test_ell_arr, check_center_vec_list, method_name, argument)

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
                    cent_vec_1_c_vec, shape_mat_1_c_vec = zip(*map(lambda ell: ell.double(),
                                                                   ell_obj_1_vec.flatten()))
                    cent_vec_2_c_vec, shape_mat_2_c_vec = zip(*map(lambda ell: ell.double(),
                                                                   ell_obj_2_vec.flatten()))
                    is_ok = np.array_equal(cent_vec_1_c_vec, cent_vec_2_c_vec) and \
                        np.array_equal(shape_mat_1_c_vec, shape_mat_2_c_vec) and \
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
