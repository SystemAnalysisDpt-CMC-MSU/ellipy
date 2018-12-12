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

    def __aux_read_file(self):
        method_name = get_caller_name_ext(2)[0]
        method_name = 'test' + method_name[5].upper() + method_name[6:]
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
