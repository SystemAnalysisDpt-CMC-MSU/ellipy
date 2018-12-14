from ellipy.elltool.core.ellipsoid.Ellipsoid import *
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
        ell_arr = cls.from_rep_mat(*args, **kwargs)
        return ell_arr

    @classmethod
    def from_dict(cls, dict_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        ell_arr = cls.from_dict(dict_arr)
        return ell_arr

    @classmethod
    def _shape_single_internal(cls, is_mod_scal: bool, mod_mat: np.ndarray):
        ell_obj = cls._shape_single_internal(is_mod_scal, mod_mat)
        return ell_obj

    @classmethod
    def _projection_single_internal(cls, ort_basis_mat: np.ndarray):
        cls._projection_single_internal(ort_basis_mat)

    @classmethod
    def _check_is_me_virtual(cls, ell_arr: Union[Iterable, np.ndarray], *args, **kwargs):
        cls._check_is_me_virtual(ell_arr, *args, **kwargs)

    @classmethod
    def _get_single_copy(cls):
        copy_ell_obj = cls._get_single_copy()
        return copy_ell_obj

    @classmethod
    def _get_scalar_polar_internal(cls, is_robust_method: bool):
        polar = cls._get_scalar_polar_internal(is_robust_method)
        return polar

    @classmethod
    def get_shape_mat(cls) -> np.ndarray:
        shape_mat = cls.get_shape_mat()
        return shape_mat

    @classmethod
    def to_dict(cls, ell_arr: Union[Iterable, np.ndarray],
                is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[np.ndarray, Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.ndarray], np.ndarray]]]:
        s_data_arr, s_field_nice_names, s_field_descr, _ = cls.to_dict(ell_arr, is_prop_included)
        return Tuple(s_data_arr, s_field_nice_names, s_field_descr)


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

        def hilb(num: int) -> np.ndarray:
            j = np.array([range(1, num+1)], dtype=np.float64)
            h = 1. / (j.T + j - 1)
            return h

        def invhilb(n: int) -> np.ndarray:
            h = np.zeros((n, n), dtype=np.float64)
            p = n
            for i in range(0, n):
                r = p * p
                h[i, i] = r / (2 * (i + 1) - 1)
                for j in range(i + 1, n):
                    r = -((n - (j + 1) + 1) * r * (n + (j + 1) - 1)) / (((j + 1) - 1) ** 2)
                    h[i, j] = r / ((i + 1) + (j + 1) - 1)
                    h[j, i] = r / ((i + 1) + (j + 1) - 1)
                p = ((n - (i + 1)) * p * (n + (i + 1))) / ((i + 1) ** 2)
            return h

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
            sh1_mat, sh2_mat = self.aux_get_test_polars(ell1)
            is_robust_better_vec[i_elem] = np.linalg.norm(exp_sh_mat - sh1_mat) <= np.linalg.norm(exp_sh_mat - sh2_mat)

            is_methods_sim_vec[i_elem] = np.linalg.norm(sh1_mat - sh2_mat) < __K_TOL

        assert np.any(is_methods_sim_vec)
        assert np.any(~is_methods_sim_vec)

        assert np.any(is_robust_better_vec)
        assert np.any(~is_robust_better_vec)

    def aux_get_test_polars(self, ell) -> Tuple[np.ndarray, np.ndarray]:
        test_ell_obj = self.get_test()
        polar1_obj = test_ell_obj.get_scalar_polar_test(ell, True)
        _, sh1_mat = Ellipsoid.double(polar1_obj)
        polar2_obj = test_ell_obj.get_scalar_polar_test(ell, False)
        _, sh2_mat = Ellipsoid.double(polar2_obj)
        return tuple((sh1_mat, sh2_mat))

    def test_negative(self):
        def run():
            ell1 = self.ellipsoid(np.ones((2, 1), dtype=np.float64), np.eye(2))
            test_ell_obj = self.get_test()
            test_ell_obj.get_scalar_polar_test(ell1, False)

        with pytest.raises(Exception) as e:
            run()
        assert 'degenerateEllipsoid' in str(e.value)
