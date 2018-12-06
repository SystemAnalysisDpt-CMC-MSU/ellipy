from ellipy.elltool.core.ellipsoid.Ellipsoid import *


class TestEllipsoidBasicSecondTC:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    def test_projection(self):
        project_mat = np.array([[1, 0], [0, 1], [0, 0]])
        cent_vec = np.array([[-2],[-1], [4]])
        shape_mat = np.array([[4, -1, 0], [-1, 1, 0], [0, 0, 9]])
        self.aux_test_projection('projection', cent_vec, shape_mat, project_mat)
        #
        project_mat = np.array([[1,0], [0, 0], [0, 1]])
        cent_vec = np.array([[2], [4], [3]])
        shape_mat = np.array([[3, 1, 1], [1, 4, 1], [1, 1, 8]])
        dim_vec = np.array([[2, 2, 3, 4]])
        self.aux_test_projection('projection', cent_vec, shape_mat, project_mat, dim_vec)
        #
        dim_vec = np.array([[0, 0, 2, 0]])
        self.aux_test_projection('projection', cent_vec, shape_mat, project_mat, dim_vec)

    def test_get_projection(self):
        project_mat = np.array([[1,0], [0, 0], [0, 1]])
        cent_vec = np.array([[2], [4], [3]])
        shape_mat = np.array([[3, 1, 1], [1, 4, 1], [1, 1, 8]])
        dim_vec = np.array([[2, 2, 3, 4]])
        self.aux_test_projection('get_projection', cent_vec, shape_mat, project_mat, dim_vec)
        #
        dim_vec = np.array([[0, 0, 2, 0]])
        self.aux_test_projection('get_projection', cent_vec, shape_mat, project_mat, dim_vec)

    @staticmethod
    def aux_test_projection(ell_factory_obj: Union[Iterable, np.ndarray], method_name: str,
                            cent_vec: np.ndarray, shape_mat: np.ndarray,
                            pr_mat: np.ndarray, dim_vec: np.ndarray):

        def isequal_internal(ell_obj_1_vec: Union[Iterable, np.ndarray],
                             ell_obj_2_vec: Union[Iterable, np.ndarray]) -> bool:
            cent_vec_1_c_vec, shape_mat_1_c_vec = zip(*map(lambda ell_obj: ell_obj.double(),
                                                           ell_obj_1_vec.flatten()))
            cent_vec_2_c_vec, shape_mat_2_c_vec = zip(*map(lambda ell_obj: ell_obj.double(),
                                                           ell_obj_2_vec.flatten()))
            is_ok = np.array_equal(cent_vec_1_c_vec, cent_vec_2_c_vec) and \
                    np.array_equal(shape_mat_1_c_vec, shape_mat_2_c_vec) and \
                    np.array_equal(ell_obj_1_vec.shape, ell_obj_2_vec.shape)
            return is_ok

        inp_obj_modify_list = ['projection']
        inp_obj_not_modify_list = ['get_projection']
        pr_cent_vec = pr_mat.T @ cent_vec
        pr_shape_mat = pr_mat.T @ shape_mat @ pr_mat
        ell_obj = ell_factory_obj.ellipsoid(cent_vec, shape_mat)
        comp_ell_obj = ell_factory_obj.ellipsoid(pr_cent_vec, pr_shape_mat)
        if method_name in inp_obj_modify_list:
            is_inp_obj_modify = True
        elif method_name in inp_obj_not_modify_list:
            ell_copy_obj = ell_obj.get_copy()
            is_inp_obj_modify = False
        else:
            throw_error('wrongInput:bad_method_name',
                      'Allowed method names: ' + inp_obj_modify_list[0] +
                      ', ' + inp_obj_not_modify_list[0] + '. Input name: ' + method_name)
        if dim_vec is None:
            pr_ell_obj = getattr(ell_obj, method_name)(pr_mat)
            test_is_right_1 = isequal_internal(comp_ell_obj, pr_ell_obj)
            if is_inp_obj_modify:
                # additional test for modification of input object
                test_is_right_2 = comp_ell_obj.is_equal(ell_obj)
            else:
                # additional test for absence of input object's modification
                test_is_right_2 = ell_copy_obj.is_equal(ell_obj)
        else:
            ell_arr = ell_obj.rep_mat(dim_vec)
            if not is_inp_obj_modify:
                ell_copy_arr = ell_copy_obj.rep_mat(dim_vec)
            pr_ell_arr = getattr(ell_arr, method_name)(pr_mat)
            comp_ell_arr = comp_ell_obj.rep_mat(dim_vec)
            test_is_right_1 = isequal_internal(comp_ell_arr, pr_ell_arr)
            if is_inp_obj_modify:
                # additional test for modification of input array
                test_is_right_2 = np.all(comp_ell_arr[:].is_equal(ell_arr[:]))
            else:
                # additional test for absence of input array's modification
                test_is_right_2 = np.all(ell_copy_arr[:].is_equal(ell_arr[:]))
        assert test_is_right_1
        assert test_is_right_2

    def test_rho(self):
        def check_rho_res(sup_arr: np.ndarray, bp_arr: np.ndarray):
            is_rho_ok = np.all(sup_arr[:] == 5)
            is_bp_ok = np.all(bp_arr[0, :] == 5) and np.all(bp_arr[1, :] == 0)
            assert is_rho_ok and is_bp_ok

        def check_rho_size(sup_arr: np.ndarray, bp_arr: np.ndarray,
                           dir_arr: np.ndarray, arr_size_vec: np.ndarray):
            is_rho_ok = np.all(sup_arr.shape == arr_size_vec)
            is_bp_ok = bp_arr.shape == dir_arr.shape
            assert is_rho_ok and is_bp_ok
        #
        dir_mat = np.array([[1, 1], [0, 0]])
        ell_obj_mat = np.diag(np.array([9, 25]))
        ell_obj_cen_vec = np.array([[2],[0]])
        ell_obj = self.ellipsoid(ell_obj_cen_vec, ell_obj_mat)
        ell_vec = [ell_obj, ell_obj, ell_obj]
        #
        # Check one ell - one dirs
        sup_val, bp_vec = rho(ell_obj, dir_mat[:, 0])
        check_rho_res(sup_val, bp_vec)
        check_rho_size(sup_val, bp_vec, np.ones((2, 1)), np.array([[1, 1]]))
        #
        # Check one ell - multiple dirs
        sup_arr, bp_mat = rho(ell_obj, dir_mat)
        check_rho_res(sup_arr, bp_mat)
        check_rho_size(sup_arr, bp_mat, dir_mat, np.array([[1, 2]]))
        #
        # Check multiple ell - one dir
        sup_arr, bp_mat = rho(ell_vec, dir_mat[:, 0])
        check_rho_res(sup_arr, bp_mat)
        check_rho_size(sup_arr, bp_mat, np.ones((2, 3)), np.array([[1, 3]]))
        #
        # Check multiple ell - multiple dirs
        arr_size_vec = np.array([2, 3, 4])
        dir_arr = np.zeros((2,) + tuple(arr_size_vec))
        dir_arr[0,:] = 1
        test_ell = self.ellipsoid(ell_obj_cen_vec, ell_obj_mat)
        ell_arr = test_ell.rep_mat(arr_size_vec)
        sup_arr, bp_arr = rho(ell_arr, dir_arr)
        check_rho_res(sup_arr, bp_arr)
        check_rho_size(sup_arr, bp_arr, dir_arr, arr_size_vec)
        #
        # Check array ell - one dir
        sup_arr, bp_arr = rho(ell_arr, dir_mat[:, 0])
        check_rho_res(sup_arr, bp_arr)
        check_rho_size(sup_arr, bp_arr, dir_arr, arr_size_vec)
        #
        # Check one ell - array dir
        sup_arr, bp_arr = rho(ell_obj, dir_arr)
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
            rho(ell2_arr, dir_arr)
        assert 'wrongInput:wrongSizes' in str(e.value)
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            rho(ell_arr, dir2_arr)
        assert 'wrongInput:wrongSizes' in str(e.value)
        ell_vec = np.array([[ell_obj, ell_obj, ell_obj]])
        dir_mat = np.eye(2)
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            rho(ell_vec, dir_mat)
        assert 'wrongInput:wrongSizes' in str(e.value)
        ell_vec = np.array([[ell_obj, ell_obj, ell_obj]]).T
        dir_mat = np.eye(2)
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            rho(ell_vec, dir_mat)
        assert 'wrongInput:wrongSizes' in str(e.value)
        ell_empt_arr = self.ellipsoid.empty([0, 0, 2, 0])
        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            rho(ell_empt_arr, dir_mat)
        assert 'wrongInput:wrongSizes' in str(e.value)

