from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.elltool.core.hyperplane.Hyperplane import *
from ellipy.gen.common.common import get_caller_name_ext
import scipy.io
import os
import pytest


class TestHyperplaneTestCase:
    test_data_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'HyperplaneTestCase')

    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def hyperplane(self, *args, **kwargs):
        return Hyperplane(*args, **kwargs)

    def test_uminus(self):
        s_inp_data = self.__aux_read_file()
        test_normal_vec = s_inp_data['testNormalVec'].flatten()
        test_constant = s_inp_data['testConstant'].flatten()
        test_hyperplane = self.hyperplane(test_normal_vec, test_constant)
        minus_test_hyraplane = -test_hyperplane
        res = self.__is_normal_and_constant_right(-test_normal_vec, -test_constant, minus_test_hyraplane)
        assert res

    def test_is_parallel(self):
        s_inp_data = self.__aux_read_file()
        test_hyperplanes_vec = s_inp_data['testHyperplanesVec'].flat[0]
        test_hyperplanes_vec = self.hyperplane(test_hyperplanes_vec[0], test_hyperplanes_vec[1])
        is_parallel_vec = s_inp_data['isParallelVec'].flatten()
        compare_hyperplanes_vec = s_inp_data['compareHyperplanesVec'].flat[0]
        compare_hyperplanes_vec = self.hyperplane(compare_hyperplanes_vec[0], compare_hyperplanes_vec[1])

        tested_is_parallel = test_hyperplanes_vec.is_parallel(
            [test_hyperplanes_vec], [compare_hyperplanes_vec])
        assert np.array_equal(tested_is_parallel, is_parallel_vec)

    def test_contains(self):
        s_inp_data = self.__aux_read_file()

        test_hyperplanes_vec = s_inp_data['testHyperplanesVec'].flatten()
        test_hyperplanes_vec = np.array([self.hyperplane(elem[0], elem[1]) for elem in test_hyperplanes_vec])
        test_vectors_mat = s_inp_data['testVectorsMat']
        is_contained_vec = s_inp_data['isContainedVec'].flatten()
        is_contained_tested_vec = test_hyperplanes_vec.flat[0].contains(test_hyperplanes_vec, test_vectors_mat)
        assert np.array_equal(is_contained_vec, is_contained_tested_vec)

        test_hyp = self.hyperplane(np.array([1, 0, 0]), [1])
        test_vectors_mat = np.array([
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        is_contained_vec = test_hyp.contains(np.array([test_hyp]), test_vectors_mat)
        is_contained_tested_vec = [True, 0, 0, 0]
        assert np.array_equal(is_contained_vec, is_contained_tested_vec)

        test_first_hyp = self.hyperplane(np.array([1, 0]), 1)
        test_sec_hyp = self.hyperplane(np.array([1, 1]), 1)
        test_third_hyp = self.hyperplane(np.array([0, 1]), 1)
        test_hyp_mat = np.array([
            [test_first_hyp, test_sec_hyp],
            [test_first_hyp, test_third_hyp]
        ])
        test_vectors = np.array([1, 0])
        is_contained_mat = test_hyp_mat.flat[0].contains(test_hyp_mat, test_vectors)
        is_contained_tested_mat = np.array([
            [True, True],
            [True, False]
        ])
        assert np.array_equal(is_contained_mat, is_contained_tested_mat)

        n_elems = 24
        test_hyp_arr = np.array([self.hyperplane(np.array([1, 1]), 2) for _ in range(n_elems)])
        test_hyp_arr = np.reshape(test_hyp_arr, newshape=(2, 3, 4))
        test_vectors_arr = np.zeros(shape=(2, 2, 3, 4))
        test_vectors_arr[:, 1, 2, 3] = np.array([1, 1])
        is_contained_arr = test_hyp_arr.flat[0].contains(test_hyp_arr, test_vectors_arr)
        is_contained_tested_arr = np.zeros(shape=(2, 3, 4), dtype=np.bool)
        is_contained_tested_arr[1, 2, 3] = True
        assert np.array_equal(is_contained_arr, is_contained_tested_arr)

    def test_hyperplane_and_double(self):
        s_inp_data = self.__aux_read_file()
        test_normal_vec = s_inp_data['testNormalVec'].flatten()
        test_const = s_inp_data['testConstant'].flatten()

        testing_hyperplane = self.hyperplane(test_normal_vec, test_const)
        assert self.__is_normal_and_constant_right(test_normal_vec, test_const, testing_hyperplane)

        test_const = [0]
        testing_hyperplane = self.hyperplane(test_normal_vec)
        assert self.__is_normal_and_constant_right(test_normal_vec, test_const, testing_hyperplane)

        test_normals_mat = s_inp_data['testNormalsMat']
        test_constant_vec = s_inp_data['testConstants'].flatten()

        testing_hyperplane_vec = np.array([self.hyperplane(x, y)
                                          for (x, y) in zip(test_normals_mat.T, test_constant_vec)])

        n_hyperplanes = np.shape(test_normals_mat)[1]
        n_res = 0
        for i_hyperplane in range(n_hyperplanes):
            n_res += self.__is_normal_and_constant_right(test_normals_mat[:, i_hyperplane],
                                                         [test_constant_vec[i_hyperplane]],
                                                         testing_hyperplane_vec[i_hyperplane])
        assert n_hyperplanes == n_res

        test_normals_mat = np.array([
            [3, 4, 43, 1],
            [1, 0, 3, 3],
            [5, 2, 2, 12]
        ])
        test_const = 2
        testing_hyperplane_vec = np.array([self.hyperplane(x, test_const) for x in test_normals_mat.T])

        n_hyperplanes = np.shape(test_normals_mat)[1]
        n_res = 0
        for i_hyperplane in range(n_hyperplanes):
            n_res += self.__is_normal_and_constant_right(test_normals_mat[:, i_hyperplane],
                                                         [test_const], testing_hyperplane_vec[i_hyperplane])
        assert n_hyperplanes == n_res

        test_norm_arr = np.ones(shape=(10, 2, 2))
        test_const_arr = 2 * np.ones(shape=(2, 2))
        test_hyp_arr = np.reshape(np.array([self.hyperplane(x, y) for (x, y) in
                                            zip(np.reshape(test_norm_arr, newshape=(10, 4)).T,
                                                np.reshape(test_const_arr, newshape=(4,)))]), newshape=(2, 2))
        is_pos = np.array_equal(np.shape(test_hyp_arr), [2, 2])
        is_pos = is_pos and self.__is_normal_and_constant_right(test_norm_arr[:, 0, 0],
                                                                [test_const_arr[0, 0]],
                                                                test_hyp_arr[0, 0])
        is_pos = is_pos and self.__is_normal_and_constant_right(test_norm_arr[:, 0, 1],
                                                                [test_const_arr[0, 1]],
                                                                test_hyp_arr[0, 1])
        assert is_pos

        test_normal_vec = np.array([3, 4, 43, 1])
        test_const = np.array([2, 3, 4, 5, 6, 7])
        n_const = np.size(test_const)
        testing_hyperplane_vec = np.array([self.hyperplane(test_normal_vec, y) for y in test_const])
        assert np.array_equal((n_const,), np.shape(testing_hyperplane_vec))

    def test_wrong_input(self):
        s_inp_data = self.__aux_read_file()
        test_constant = s_inp_data['testConstant']
        test_hyperplane = s_inp_data['testHyperplane'].flat[0]
        test_hyperplane = self.hyperplane(test_hyperplane[0], test_hyperplane[1])
        nan_vec = s_inp_data['nanVector']
        inf_vec = s_inp_data['infVector']

        with pytest.raises(Exception) as e:
            # noinspection PyChecker
            test_hyperplane.contains(np.array([test_hyperplane]), nan_vec)
            assert 'wrongInput' in str(e.value)
            self.hyperplane(inf_vec, test_constant)
            assert 'wrongInput' in str(e.value)
            self.hyperplane(nan_vec, test_constant)
            assert 'wrongInput' in str(e.value)

        return self

    def test_get_abs_tol(self):
        norm_vec = np.ones((3, 1))
        const = 0
        test_abs_tol = 1.
        args = [norm_vec, const]
        kwargs = {'abs_tol': test_abs_tol}
        h_plane_arr = np.zeros(shape=(2, 2, 2), dtype=object)
        h_plane_arr[:][:][0] = np.array([[self.hyperplane(*args, **kwargs), self.hyperplane(*args, **kwargs)],
                                        [self.hyperplane(*args, **kwargs), self.hyperplane(*args, **kwargs)]])
        h_plane_arr[:][:][1] = np.array([[self.hyperplane(*args, **kwargs), self.hyperplane(*args, **kwargs)],
                                        [self.hyperplane(*args, **kwargs), self.hyperplane(*args, **kwargs)]])
        size_arr = np.shape(h_plane_arr)
        test_abs_tol_arr = np.tile(test_abs_tol, size_arr)
        h_plane_arr_abs = ABasicEllipsoid.get_abs_tol(h_plane_arr, f_prop_fun=None)
        is_ok_arr = (np.equal(test_abs_tol_arr.flatten(), h_plane_arr_abs))
        is_ok = np.all(is_ok_arr)
        assert is_ok

    def test_rel_tol(self):
        def aux_test_rel_tol(hp, rel_tol):
            assert ABasicEllipsoid.get_rel_tol(hp, f_prop_fun=None) == rel_tol
        hp = self.hyperplane()
        aux_test_rel_tol(hp, 1e-5)

        args = [np.ones(1), 1]
        kwargs = {'rel_tol': 1e-3}
        hp = self.hyperplane(*args, **kwargs)
        aux_test_rel_tol(hp, 1e-3)

    def __aux_read_file(self):
        method_name = get_caller_name_ext(2)[0]
        method_split = [name[0].upper() + name[1:] for name in method_name.split('_')[1:]]
        method_name = 'test' + ''.join(method_split)
        inp_file_name = os.path.join(self.test_data_root_dir, method_name + '_inp.mat')
        #
        return scipy.io.loadmat(inp_file_name)

    @staticmethod
    def __is_normal_and_constant_right(test_normal, test_constant, testing_hyperplane):
        result_normal, result_constant = testing_hyperplane.double()
        result_constant = np.array(result_constant).flatten()

        test_norm_size_vec = np.shape(test_normal)
        res_norm_size_vec = np.shape(result_normal)

        if np.array_equal(test_norm_size_vec, res_norm_size_vec):
            return np.array_equal(test_normal, result_normal) and np.array_equal(test_constant, result_constant)
        else:
            return False
