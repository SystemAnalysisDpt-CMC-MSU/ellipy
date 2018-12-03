from ellipy.elltool.core.abasicellipsoid.ABasicEllipsoid import *
from ellipy.gras.la.la import is_mat_pos_def
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Iterable
import numpy as np
from numpy import matlib as ml


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
            '_n_plot_3d_points': '_n_plot_3d_points',
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
        if type(center_vec) != np.ndarray or not is_numeric(center_vec):
            throw_error('wrongInput:center_vec', 'center_vec should be numeric array')
        center_vec = np.array(try_treat_as_real(center_vec, self._abs_tol), dtype=np.float64)
        if center_vec.size != np.max(center_vec.shape):
            throw_error('wrongInput:center_vec', 'center_vec should be a vector')
        center_vec = center_vec.flatten()
        if not np.all(np.isfinite(center_vec)):
            throw_error('wrongInput:center_vec', 'center_vec should have all finite values')
        self.__center_vec = np.copy(center_vec)

    @classmethod
    def projection(cls, ell_arr: Union[Iterable, np.ndarray], basis_mat: np.ndarray) -> np.ndarray:
        cls._check_is_me_virtual(ell_arr)
        if not is_numeric(basis_mat):
            throw_error('wrongInput:basis_mat',
                        'second input argument must be matrix with orthogonal columns')
        if not np.any(cls.is_empty(ell_arr).flatten()):
            n_dim, n_basis = basis_mat.shape
            n_dims_arr = cls.dimension(ell_arr)
            if not (n_basis <= n_dim and np.all(n_dims_arr.flatten(1) == n_dim)):
                throw_error('wrongInput', 'dimensions mismatch or number of basis vectors too large');
            # check the orthogonality of the columns of basis_mat
            scal_prod_mat = basis_mat.T @ basis_mat
            norm_sq_vec = np.diag(scal_prod_mat)
            _, abs_tol = cls.get_abs_tol([cls, ell_arr], lambda z: np.max(z))
            is_ortogonal_mat = (scal_prod_mat - diag(norm_sq_vec)) > abs_tol
            if np.any(is_ortogonal_mat.flatten(1)):
                throw_error('wrongInput','basis vectors must be orthogonal');
            # normalize the basis vectors
            norm_mat = ml.repmat(np.sqrt(norm_sq_vec.T), n_dim, 1)
            ort_basis_mat = basis_mat / norm_mat
            # compute projection
            ell_arr = np.array(map(lambda x: projection_single_internal(x, ort_basis_mat), ell_arr.flatten()))
        return ell_arr

    def get_center_vec(self):
        return self._center_vec

    @classmethod
    def get_n_plot_2d_points(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        return cls.get_property(ell_arr, '_n_plot_2d_points', None)

    @classmethod
    def get_n_plot_3d_points(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        return cls.get_property(ell_arr, '_n_plot_3d_points', None)

    @classmethod
    def mtimes(cls, mult_mat: np.ndarray, inp_ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def __rmul__(self, mult_mat: np.ndarray):
        return self.mtimes(mult_mat, [self]).flatten()[0]

    @classmethod
    def shape(cls, ell_arr: Union[Iterable, np.ndarray], mod_mat: np.ndarray) -> np.ndarray:
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
    def from_dict(cls, dict_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def is_degenerate(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
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
    def volume(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def dimension(cls, ell_arr: Union[Iterable, np.ndarray], return_rank=False) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        cls._check_is_me_virtual(ell_arr)
        ell_arr = np.array(ell_arr)
        ell_list = list(ell_arr.flatten())
        ell_shape_vec = ell_arr.shape
        if return_rank:
            ndim_list, rank_list = zip(*[
                (np.size(ell_obj.get_center_vec()), np.linalg.matrix_rank(ell_obj.get_shape_mat()))
                for ell_obj in ell_list
            ])
            return np.reshape(np.array(ndim_list), ell_shape_vec), np.reshape(np.array(rank_list), ell_shape_vec)
        else:
            ndim_list = [np.size(ell_obj.get_center_vec()) for ell_obj in ell_list]
            return np.reshape(np.array(ndim_list), ell_shape_vec)

    @classmethod
    def get_shape(cls, ell_arr: Union[Iterable, np.ndarray], mod_mat: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def min_eig(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def max_eig(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def plus(cls, *args) -> np.ndarray:
        pass

    def __add__(self, b_vec: np.ndarray):
        return self.plus([self], b_vec).flatten()[0]

    def __radd__(self, b_vec: np.ndarray):
        return self.plus(b_vec, [self]).flatten()[0]

    @classmethod
    def minus(cls, *args) -> np.ndarray:
        pass

    def __sub__(self, b_vec: np.ndarray):
        return self.minus([self], b_vec).flatten()[0]

    def __rsub__(self, b_vec: np.ndarray):
        return self.minus(b_vec, [self]).flatten()[0]

    @classmethod
    def trace(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass
