from ellipy.elltool.core.abasicellipsoid.ABasicEllipsoid import *
from ellipy.gras.la.la import is_mat_pos_def
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict
import numpy as np


class AEllipsoid(ABasicEllipsoid, ABC):
    @classmethod
    def _get_prop_aliases_dict(cls) -> Dict[str, str]:
        aliases_dict = ABasicEllipsoid._get_prop_aliases_dict()
        aliases_dict.update({
            'nPlot2dPoints': '_n_plot_2d_points',
            'n_plot_2d_points': '_n_plot_2d_points',
            '_n_plot_2d_points': '_n_plot_2d_points',
            'nPlot3dPoints': '_n_plot_3d_points',
            'n_plot_3d_points': '_n_plot_3d_points',
            '_n_plot_3d_points': '_n_plot_2d_points',
            'centerVec': '_center_vec',
            'center_vec': '_center_vec',
            '_center_vec': '_center_vec'})
        return aliases_dict

    def __init__(self):
        ABasicEllipsoid.__init__(self)

    @property
    def _center_vec(self) -> np.ndarray:
        return np.copy(self.__center_vec)

    @_center_vec.setter
    def _center_vec(self, center_vec) -> None:
        self.__center_vec = np.copy(center_vec)

    @classmethod
    def projection(cls, ell_arr: np.ndarray, basis_mat: np.ndarray) -> np.ndarray:
        pass

    def get_center_vec(self):
        return self._center_vec

    @classmethod
    def get_n_plot_2d_points(cls, ell_arr: np.ndarray) -> np.ndarray:
        return cls.get_property(ell_arr, '_n_plot_2d_points', None)

    @classmethod
    def get_n_plot_3d_points(cls, ell_arr: np.ndarray) -> np.ndarray:
        return cls.get_property(ell_arr, '_n_plot_2d_points', None)

    @classmethod
    def mtimes(cls, mult_mat: np.ndarray, inp_ell_arr: np.ndarray) -> np.ndarray:
        pass

    def __rmul__(self, mult_mat: np.ndarray):
        return self.mtimes(mult_mat, np.array([self])).flatten()[0]

    @classmethod
    def shape(cls, ell_arr: np.ndarray, mod_mat: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_shape_mat(self):
        pass

    @abstractmethod
    def _get_scalar_polar_internal(self, is_robust_method: bool):
        pass

    @abstractmethod
    def _shape_single_internal(self, is_mod_scal: bool, mod_mat: np.ndarray):
        pass

    @classmethod
    @abstractmethod
    def from_rep_mat(cls, *args, **kwargs) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, dict_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def is_degenerate(cls, ell_arr: np.ndarray) -> np.ndarray:
        cls._check_is_me_virtual(ell_arr)
        ell_arr = np.array(ell_arr)
        if np.any(cls.is_empty(ell_arr).flatten()):
            throw_error('wrongInput:emptyEllipsoid', 'input argument contains empty ellipsoid')
        if ell_arr.size == 0:
            return np.ones(ell_arr.shape, dtype=bool)
        else:
            # noinspection PyProtectedMember
            return ~np.reshape(
                np.array([is_mat_pos_def(ell_obj.get_shape_mat(), ell_obj._abs_tol)
                          for ell_obj in list(ell_arr.flatten())]), ell_arr.shape)

    @classmethod
    def volume(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def dimension(cls, ell_arr: np.ndarray, return_rank=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        cls._check_is_me_virtual(ell_arr)
        ell_list = list(ell_arr.flatten())
        ell_shape_vec = ell_arr.shape
        if return_rank:
            ndim_list, rank_list = zip(*[
                (np.size(ell_obj.get_center_vec()), np.rank(ell_obj.get_shape_mat()))
                for ell_obj in ell_list
            ])
            return np.reshape(np.array(ndim_list), ell_shape_vec), np.reshape(np.array(rank_list), ell_shape_vec)
        else:
            ndim_list = [np.size(ell_obj.get_center_vec()) for ell_obj in ell_list]
            return np.reshape(np.array(ndim_list), ell_shape_vec)

    @classmethod
    def get_shape(cls, ell_arr: np.ndarray, mod_mat: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def min_eig(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def max_eig(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def plus(cls, *args) -> np.ndarray:
        pass

    def __add__(self, b_vec: np.ndarray):
        return self.plus(np.array([self]), b_vec).flatten()[0]

    def __radd__(self, b_vec: np.ndarray):
        return self.plus(b_vec, np.array([self])).flatten()[0]

    @classmethod
    def minus(cls, *args) -> np.ndarray:
        pass

    def __sub__(self, b_vec: np.ndarray):
        return self.minus(np.array([self]), b_vec).flatten()[0]

    def __rsub__(self, b_vec: np.ndarray):
        return self.minus(b_vec, np.array([self])).flatten()[0]

    @classmethod
    def trace(cls, ell_arr: np.ndarray) -> np.ndarray:
        pass
