from ellipy.elltool.core.aellipsoid.AEllipsoid import *
from ellipy.gen.common.common import throw_error
from typing import Tuple, Dict, Callable
import numpy as np


class Ellipsoid(AEllipsoid):
    @classmethod
    def _get_prop_aliases_dict(cls) -> Dict[str, str]:
        aliases_dict = AEllipsoid._get_prop_aliases_dict()
        aliases_dict.update({
            'shapeMat': '_shape_mat',
            'shape_mat': '_shape_mat',
            '_shape_mat': '_shape_mat'})
        return aliases_dict

    def __init__(self, *args, **kwargs):
        AEllipsoid.__init__(self)
        # write

    @property
    def _shape_mat(self) -> np.ndarray:
        return self.__shape_mat

    @_shape_mat.setter
    def _shape_mat(self, shape_mat: np.ndarray) -> None:
        if shape_mat.ndim != 2:
            throw_error('wrongInput:shape_mat', 'shape_mat must be a matrix')
        if shape_mat.shape[0] != shape_mat.shape[1]:
            throw_error('wrongInput:shape_mat', 'shape_mat must be a square matrix')
        if np.any(np.isnan(shape_mat.flatten())):
            throw_error('wrongInput:shape_mat', 'configuration matrix cannot contain NaN values')
        self.__shape_mat = shape_mat

    def quad_func(self) -> np.ndarray:
        pass

    def get_shape_mat(self) -> np.ndarray:
        return np.copy(self._shape_mat)

    def _get_scalar_polar_internal(self, is_robust_method: bool):
        pass

    @classmethod
    def from_rep_mat(cls, *args, **kwargs) -> np.ndarray:
        # write
        pass

    @classmethod
    def from_dict(cls, dict_arr: np.ndarray) -> np.ndarray:
        # write
        pass

    @staticmethod
    def _regularize(q_mat: np.ndarray, abs_tol: float) -> np.ndarray:
        # write
        pass

    @staticmethod
    def _rm_bad_directions(q1_mat: np.ndarray, q2_mat: np.ndarray, dirs_mat: np.ndarray, abs_tol: float) -> np.ndarray:
        pass

    @staticmethod
    def _is_bad_direction_mat(q1_mat: np.ndarray, q2_mat: np.ndarray,
                              dirs_mat: np.ndarray, abs_tol: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def is_bad_direction(self, sec_ell, dirs_mat: np.ndarray, abs_tol: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def _calc_diff_one_dir(first_ell, sec_ell, l_mat: np.ndarray,
                           p_univ_vec: np.ndarray, is_good_dir_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def _ellbndr_2dmat(n_points: int,
                       cent_vec: np.ndarray, q_mat: np.ndarray, abs_tol: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def _ellbndr_3dmat(n_points: int,
                       cent_vec: np.ndarray, q_mat: np.ndarray, abs_tol: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def __get_grid_by_factor(self, factor_vec: np.ndarray):
        pass

    @classmethod
    def _check_is_me(cls, ell_arr, *args, **kwargs):
        # write
        pass

    def _shape_single_internal(self, is_mod_scal: bool, mod_mat: np.ndarray):
        pass

    def _projection_single_internal(self, ort_basis_mat: np.ndarray):
        pass

    @classmethod
    def _check_is_me_virtual(cls, ell_arr: np.ndarray, *args, **kwargs):
        cls._check_is_me(ell_arr, *args, **kwargs)

    def _get_single_copy(self):
        # write
        pass

    def double(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._center_vec, self._shape_mat

    def ellbndr_2d(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def ellbndr_3d(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @classmethod
    def ge(cls, first_ell_arr: np.ndarray, second_ell_arr: np.ndarray) -> np.ndarray:
        pass

    def __ge__(self, other):
        return self.ge(np.array([self]), np.array([other])).flatten()[0]

    @classmethod
    def gt(cls, first_ell_arr: np.ndarray, second_ell_arr: np.ndarray) -> np.ndarray:
        pass

    def __gt__(self, other):
        return self.gt(np.array([self]), np.array([other])).flatten()[0]

    @classmethod
    def le(cls, first_ell_arr: np.ndarray, second_ell_arr: np.ndarray) -> np.ndarray:
        pass

    def __le__(self, other):
        return self.le(np.array([self]), np.array([other])).flatten()[0]

    @classmethod
    def lt(cls, first_ell_arr: np.ndarray, second_ell_arr: np.ndarray) -> np.ndarray:
        pass

    def __lt__(self, other):
        return self.lt(np.array([self]), np.array([other])).flatten()[0]

    @classmethod
    def ne(cls, first_ell_arr: np.ndarray, second_ell_arr: np.ndarray) -> Tuple[np.ndarray, str]:
        pass

    def __ne__(self, other):
        is_ne, _ = self.ne(np.array([self]), np.array([other]))
        return np.array(is_ne).flatten()[0]

    def get_boundary(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_boundary_by_factor(self, factor_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @classmethod
    def get_inv(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def get_move_2_origin(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def get_projection(cls, ell_arr: np.ndarray, basis_mat: np.ndarray) -> np.ndarray:
        pass

    def get_rho_boundary(self, n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def get_rho_boundary_by_factor(self, factor_vec: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @classmethod
    def inv(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    def is_bigger(self, sec_ell) -> bool:
        self._check_is_me(sec_ell, 'second')
        # write
        return False

    @classmethod
    def is_internal(cls, ell_arr: np.ndarray, mat_of_vec_mat: np.ndarray, mode: str) -> np.ndarray:
        pass

    @classmethod
    def minksum_ea(cls, ell_arr: np.ndarray, dir_mat: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def minksum_ia(cls, ell_arr: np.ndarray, dir_mat: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def move_2_origin(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    def parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.double()

    @classmethod
    def polar(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def rho(cls, ell_arr: np.ndarray, dirs_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @classmethod
    def to_dict(cls, ell_arr: np.ndarray, is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[List[dict], Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.ndarray], np.ndarray]]]:
        # write
        pass

    @classmethod
    def uminus(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    def __neg__(self):
        return self.uminus(np.array([self])).flatten()[0]

    def __str__(self):
        pass

    def __repr__(self):
        pass
