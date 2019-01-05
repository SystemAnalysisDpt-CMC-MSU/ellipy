from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from scipy.linalg import hilbert as hilb
from scipy.linalg import invhilbert as invhilb
import numpy as np
import pytest


class PolarEllipsoidTest(AEllipsoid):
    def __init__(self):
        AEllipsoid.__init__(self)

    @classmethod
    def get_scalar_polar_test(cls, ell: AEllipsoid, is_robust_method: bool):
        polar_obj = ell._get_scalar_polar_internal(is_robust_method)
        return polar_obj

    @classmethod
    def from_rep_mat(cls, *args, **kwargs) -> np.ndarray:
        pass

    @classmethod
    def from_dict(cls, dict_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def _shape_single_internal(cls, is_mod_scal: bool, mod_mat: np.ndarray):
        pass

    @classmethod
    def _projection_single_internal(cls, ort_basis_mat: np.ndarray):
        pass

    @classmethod
    def _check_is_me_virtual(cls, ell_arr: Union[Iterable, np.ndarray], *args, **kwargs):
        pass

    @classmethod
    def _get_single_copy(cls):
        pass

    @classmethod
    def _get_scalar_polar_internal(cls, is_robust_method: bool):
        pass

    @classmethod
    def get_shape_mat(cls) -> np.ndarray:
        pass

    @classmethod
    def to_dict(cls, ell_arr: Union[Iterable, np.ndarray],
                is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[np.ndarray, Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.ndarray], np.ndarray]]]:
        pass


class TestPolarIllCondTC:
    __ABS_TOL = 1e-8

    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    @classmethod
    def get_test(cls):
        test_ell_obj = PolarEllipsoidTest()
        return test_ell_obj

    def test_get_scalar_polar(self):
        __K_TOL = 1e-2
        __DIM_VEC = np.array(range(2, 12))
        isn_overflow_vec = np.full((__DIM_VEC.size, 1), True)
        for k in range(0, __DIM_VEC.size):
            isn_overflow_vec[k] = np.min(np.linalg.eig(np.linalg.inv(hilb(__DIM_VEC[k])))[0]) > 0
        dim_vec = __DIM_VEC[isn_overflow_vec.flatten()]
        __N_TESTS = np.size(dim_vec)

        is_robust_better_vec = np.zeros((__N_TESTS, 1), dtype=bool)
        is_methods_sim_vec = np.zeros((__N_TESTS, 1), dtype=bool)

        for i_elem in range(0, __N_TESTS):
            __N_DIMS = dim_vec[i_elem]
            sh_mat = hilb(__N_DIMS)
            exp_sh_mat = invhilb(__N_DIMS)
            ell1 = self.ellipsoid(sh_mat)
            sh1_mat, sh2_mat = self.__aux_get_test_polars(ell1)
            is_robust_better_vec[i_elem] = np.linalg.norm(exp_sh_mat - sh1_mat) <= \
                np.linalg.norm(exp_sh_mat - sh2_mat)

            is_methods_sim_vec[i_elem] = np.linalg.norm(sh1_mat - sh2_mat) < __K_TOL

        assert np.any(is_methods_sim_vec)
        assert np.any(~is_methods_sim_vec)

        assert np.any(is_robust_better_vec)
        assert np.any(~is_robust_better_vec)

    def __aux_get_test_polars(self, ell) -> Tuple[np.ndarray, np.ndarray]:
        test_ell_obj = self.get_test()
        polar1_obj = test_ell_obj.get_scalar_polar_test(ell, True)
        _, sh1_mat = polar1_obj.double()
        polar2_obj = test_ell_obj.get_scalar_polar_test(ell, False)
        _, sh2_mat = polar2_obj.double()
        return tuple((sh1_mat, sh2_mat))

    def test_negative(self):
        def run():
            ell1 = self.ellipsoid(np.ones((2, 1), dtype=np.float64), np.eye(2))
            test_ell_obj = self.get_test()
            test_ell_obj.get_scalar_polar_test(ell1, False)

        with pytest.raises(Exception) as e:
            run()
        assert 'degenerateEllipsoid' in str(e.value)
