from ellipy.elltool.core.abasicellipsoid.ABasicEllipsoid import *
from typing import Tuple, Dict, Callable
import numpy as np


class Hyperplane(ABasicEllipsoid):
    @classmethod
    def _get_prop_aliases_dict(cls) -> Dict[str, str]:
        aliases_dict = ABasicEllipsoid._get_prop_aliases_dict()
        aliases_dict.update({
            'normalVec': '_normal_vec',
            'normal_vec': '_normal_vec',
            '_normal_vec': '_normal_vec',
            'normal': '_normal_vec',
            '_normal': '_normal_vec',
            'shift': '_shift',
            '_shift': '_shift'})
        return aliases_dict

    def __init__(self, *args, **kwargs):
        ABasicEllipsoid.__init__(self)

    @property
    def _normal_vec(self) -> np.ndarray:
        return np.copy(self.__normal_vec)

    @_normal_vec.setter
    def _normal_vec(self, normal_vec: np.ndarray):
        self.__normal_vec = normal_vec

    @property
    def _shift(self) -> np.float64:
        return self.__shift

    @_shift.setter
    def _shift(self, shift: np.float64):
        self.__shift = shift

    @classmethod
    def from_rep_mat(cls, *args, **kwargs) -> np.ndarray:
        # write
        pass

    @classmethod
    def from_dict(cls, dict_arr: np.ndarray) -> np.ndarray:
        # write
        pass

    @classmethod
    def _check_is_me(cls, ell_arr, *args, **kwargs):
        cls._check_is_me_internal(ell_arr, *args, **kwargs)

    @classmethod
    def _check_is_me_virtual(cls, ell_arr: np.ndarray, *args, **kwargs):
        cls._check_is_me(ell_arr, *args, **kwargs)

    def _get_single_copy(self):
        # write
        pass

    @classmethod
    def contains(cls, hyp_arr: np.ndarray, x_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def dimension(cls, ell_arr: np.ndarray, return_rank=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # write
        pass

    def double(self) -> Tuple[np.ndarray, np.float64]:
        return self._normal_vec, self._shift

    def parameters(self) -> Tuple[np.ndarray, np.float64]:
        return self.double()

    @classmethod
    def is_parallel(cls, first_hyp_arr: np.ndarray, sec_hyp_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def ne(cls, first_ell_arr: np.ndarray, second_ell_arr: np.ndarray) -> Tuple[np.ndarray, str]:
        pass

    def __ne__(self, other):
        is_ne, _ = self.ne(np.array([self]), np.array([other]))
        return np.array(is_ne).flatten()[0]

    @classmethod
    def to_dict(cls, hyp_arr: np.ndarray, is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[List[dict], Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.ndarray], np.ndarray]]]:
        # write
        pass

    @classmethod
    def uminus(cls, hyp_arr: np.ndarray) -> np.ndarray:
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass
