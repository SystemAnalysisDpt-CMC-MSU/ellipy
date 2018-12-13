from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.elltool.core.hyperplane.Hyperplane import *
from ellipy.gen.common.common import get_caller_name_ext
import scipy.io
import os


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
        test_hyraplane = self.hyperplane(test_normal_vec, test_constant)
        minus_test_hyraplane = -test_hyraplane
        res = self.__is_normal_and_constant_right(-test_normal_vec, -test_constant, minus_test_hyraplane)
        assert res

        return self

    def test_is_parallel(self):
        s_inp_data = self.__aux_read_file()
        test_hyperplanes_vec = s_inp_data['testHyperplanesVec'].flatten()[0]
        test_hyperplanes_vec = Hyperplane(test_hyperplanes_vec[0], test_hyperplanes_vec[1])
        is_parallel_vec = s_inp_data['isParallelVec'].flatten()
        compare_hyperplanes_vec = s_inp_data['compareHyperplanesVec'].flatten()[0]
        compare_hyperplanes_vec = Hyperplane(compare_hyperplanes_vec[0], compare_hyperplanes_vec[1])

        tested_is_parallel = Hyperplane.is_parallel([test_hyperplanes_vec], [compare_hyperplanes_vec])
        assert np.array_equal(tested_is_parallel, is_parallel_vec)

        return self

    def test_contains(self):
        s_inp_data = self.__aux_read_file()

        test_hyperplanes_vec = s_inp_data['testHyperplanesVec'].flatten()
        test_hyperplanes_vec = np.array([self.hyperplane(elem[0], elem[1]) for elem in test_hyperplanes_vec])
        test_vectors_mat = s_inp_data['testVectorsMat']
        is_contained_vec = s_inp_data['isContainedVec']
        is_contained_tested_vec = Hyperplane.contains(test_hyperplanes_vec, test_vectors_mat)
        assert np.array_equal(is_contained_vec, is_contained_tested_vec)

        test_hyp = self.hyperplane([1, 0, 0], 1)
        test_vectors_mat = np.array([
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        is_contained_vec = Hyperplane.contains(np.array([test_hyp]), test_vectors_mat)
        is_contained_tested_vec = [True, 0, 0, 0]
        assert np.array_equal(is_contained_vec, is_contained_tested_vec)

        test_first_hyp = self.hyperplane([1, 0], 1)
        test_sec_hyp = self.hyperplane([1, 1], 1)
        test_third_hyp = self.hyperplane([0, 1], 1)
        test_hyp_mat = np.array([
            [test_first_hyp, test_sec_hyp],
            [test_first_hyp, test_third_hyp]
        ])
        test_vectors = np.array([1, 0])
        is_contained_mat = Hyperplane.contains(test_hyp_mat, test_vectors)
        is_contained_tested_mat = np.array([
            [True, False],
            [True, False]
        ])
        assert np.array_equal(is_contained_mat, is_contained_tested_mat)

        n_elems = 24
        test_hyp_arr = np.array([self.hyperplane([1, 1], 1) for _ in range(n_elems)])
        test_hyp_arr = np.reshape(test_hyp_arr, newshape=(2, 3, 4))
        test_vectors_arr = np.zeros(shape=(2, 2, 3, 4))
        test_vectors_arr[:, 1, 2, 3] = [1, 1]
        is_contained_arr = Hyperplane.contains(test_hyp_arr, test_vectors_arr)
        is_contained_tested_arr = np.zeros(shape=(2, 3, 4), dtype=np.bool)
        is_contained_tested_arr[1, 2, 3] = True
        assert np.array_equal(is_contained_arr, is_contained_tested_arr)

        return self

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

        testing_hyraplane_vec = np.array([self.hyperplane(x, y)
                                          for (x, y) in zip(test_normals_mat.T, test_constant_vec)])

        n_hyperplanes = np.shape(test_normals_mat)[1]
        n_res = 0
        for i_hyperplane in range(n_hyperplanes):
            n_res += self.__is_normal_and_constant_right(test_normals_mat[:, i_hyperplane],
                                                         [test_constant_vec[i_hyperplane]],
                                                         testing_hyraplane_vec[i_hyperplane])
        assert n_hyperplanes == n_res

        test_normals_mat = np.array([
            [3, 4, 43, 1],
            [1, 0, 3, 3],
            [5, 2, 2, 12]
        ])
        test_const = 2
        testing_hyraplane_vec = np.array([self.hyperplane(x, test_const) for x in test_normals_mat.T])

        n_hyperplanes = np.shape(test_normals_mat)[1]
        n_res = 0
        for i_hyperplane in range(n_hyperplanes):
            n_res += self.__is_normal_and_constant_right(test_normals_mat[:, i_hyperplane],
                                                         [test_const], testing_hyraplane_vec[i_hyperplane])
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
        testing_hyraplane_vec = np.array([self.hyperplane(test_normal_vec, y) for y in test_const])
        assert np.array_equal((n_const,), np.shape(testing_hyraplane_vec))

        return self

    def test_wrong_input(self):
        s_inp_data = self.__aux_read_file()
        test_constant = s_inp_data['testConstant']
        test_hyperplane = s_inp_data['testHyperplane'].flatten()[0]
        test_hyperplane = self.hyperplane(test_hyperplane[0], test_hyperplane[1])
        nan_vec = s_inp_data['nanVector']
        inf_vec = s_inp_data['infVector']

        # TO DO: run_and_check_error
        # TO DO: run_and_check_error
        # TO DO: run_and_check_error

    def __aux_read_file(self):
        method_name = get_caller_name_ext(2)[0]
        method_split = [name[0].upper() + name[1:] for name in method_name.split('_')[1:]]
        method_name = 'test' + ''.join(method_split)
        inp_file_name = os.path.join(self.test_data_root_dir, method_name + '_inp.mat')
        #
        return scipy.io.loadmat(inp_file_name)

    @staticmethod
    def __is_normal_and_constant_right(test_normal, test_constant, testing_hyraplane):
        if type(testing_hyraplane) != Hyperplane:
            result_normal = [elem._normal_vec.flatten() for elem in testing_hyraplane]
            result_constant = [elem._shift.flatten() for elem in testing_hyraplane]
        else:
            result_normal = testing_hyraplane._normal_vec.flatten()
            result_constant = testing_hyraplane._shift.flatten()

        test_norm_size_vec = np.shape(test_normal)
        res_norm_size_vec = np.shape(result_normal)

        if np.array_equal(test_norm_size_vec, res_norm_size_vec):
            return np.array_equal(test_normal, result_normal) and np.array_equal(test_constant, result_constant)
        else:
            return False
