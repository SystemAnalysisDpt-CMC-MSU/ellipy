from ellipy.elltool.core.core import *
import scipy.io
import numpy as np
import os


class TestCore:
    __TEST_DATA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def test_fusion_lambda(self):
        __FUSIONLAMBDA_VEC_DATA_OUTPUT = \
            scipy.io.loadmat(os.path.join(self.__TEST_DATA_ROOT_DIR, 'fusionlambda_data.mat'))['output']

        result0 = ell_fusion_lambda(0.5, np.array([[1, 1]], dtype=np.float64).T,
                                    np.array([[1, 1], [1, 1]], dtype=np.float64),
                                    np.array([[1, 1]], dtype=np.float64).T,
                                    np.array([[1, 1], [1, 0]], dtype=np.float64), 5)
        result1 = ell_fusion_lambda(0.5, np.array([[1, 1, 1]], dtype=np.float64).T,
                                    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64),
                                    np.array([[1, 1, 5]], dtype=np.float64).T,
                                    np.array([[1, 2, 1], [1, 1, 1], [1, 1, 0]], dtype=np.float64), 5)
        result2 = ell_fusion_lambda(0.5, np.array([[1, 0.00001]], dtype=np.float64).T,
                                    np.array([[1, 1], [1, 1]], dtype=np.float64),
                                    np.array([[1, 1]], dtype=np.float64).T,
                                    np.array([[1, 1], [0.00001, 0]], dtype=np.float64), 5)
        result3 = ell_fusion_lambda(1, np.array([[5, 1]], dtype=np.float64).T,
                                    np.array([[1, 1], [0, 1]], dtype=np.float64),
                                    np.array([[1, 1]], dtype=np.float64).T,
                                    np.array([[0, 2], [1, 1]], dtype=np.float64), 10)
        result4 = ell_fusion_lambda(0.5, np.array([[1, 1]], dtype=np.float64).T,
                                    np.array([[1, 1], [1, 1]], dtype=np.float64),
                                    np.array([[1, 1]], dtype=np.float64).T,
                                    np.array([[100, 1], [1, 0]], dtype=np.float64), 2)

        assert np.isclose(result0, __FUSIONLAMBDA_VEC_DATA_OUTPUT[0][0], rtol=1e-9)
        assert np.isclose(result1, __FUSIONLAMBDA_VEC_DATA_OUTPUT[0][1], rtol=1e-9)
        assert np.isclose(result2, __FUSIONLAMBDA_VEC_DATA_OUTPUT[0][2], rtol=1e-9)
        assert np.isclose(result3, __FUSIONLAMBDA_VEC_DATA_OUTPUT[0][3], rtol=1e-9)
        assert np.isclose(result4, __FUSIONLAMBDA_VEC_DATA_OUTPUT[0][4], rtol=1e-9)

    def test_valign(self):
        __VALIGN_VEC_DATA_OUTPUT = \
            scipy.io.loadmat(os.path.join(self.__TEST_DATA_ROOT_DIR, 'valign_data.mat'))['output']

        v = np.array([[1, 1]], dtype=np.float64).T
        x = np.array([[1, 1]], dtype=np.float64).T
        mat0 = ell_valign(v, x)
        v = np.array([[1.11, 1, 10]], dtype=np.float64).T
        x = np.array([[0.001, 1, 10]], dtype=np.float64).T
        mat1 = ell_valign(v, x)
        v = np.array([[1.123, 0.001]], dtype=np.float64).T
        x = np.array([[100, 1.0123]], dtype=np.float64).T
        mat2 = ell_valign(v, x)

        assert np.allclose(mat0, __VALIGN_VEC_DATA_OUTPUT[0][0], rtol=1e-9)
        assert np.allclose(mat1, __VALIGN_VEC_DATA_OUTPUT[0][1], rtol=1e-9)
        assert np.allclose(mat2, __VALIGN_VEC_DATA_OUTPUT[0][2], rtol=1e-9)
