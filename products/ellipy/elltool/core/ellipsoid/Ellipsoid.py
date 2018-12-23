from ellipy.elltool.core.aellipsoid.AEllipsoid import *
from ellipy.elltool.conf.properties.Properties import Properties
from ellipy.gras.la.la import is_mat_pos_def, is_mat_symm, try_treat_as_real, sqrtm_pos
from ellipy.gen.common.common import throw_error
from ellipy.gen.logging.logging import get_logger
from ellipy.gras.geom.ell.ell import rho_mat
from typing import Tuple, Dict, Callable, Any
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
        prop_list, _ = Properties.parse_prop(kwargs, ['abs_tol', 'rel_tol', 'n_plot_2d_points', 'n_plot_3d_points'])
        self._abs_tol = prop_list[0]
        self._rel_tol = prop_list[1]
        self._n_plot_2d_points = prop_list[2]
        self._n_plot_3d_points = prop_list[3]
        if len(args) == 0:
            self._center_vec = np.zeros((0,), dtype=np.float64)
            self._shape_mat = np.zeros((0, 0), dtype=np.float64)
        elif len(args) == 1:
            self._shape_mat = args[0]
            self._center_vec = np.zeros((self._shape_mat.shape[0],), dtype=np.float64)
        else:
            self._center_vec = args[0]
            self._shape_mat = args[1]
            if self._center_vec.size != self._shape_mat.shape[0]:
                throw_error('wrongInput', 'dimensions of center_vec and shape_mat must agree')

    @property
    def _shape_mat(self) -> np.ndarray:
        return np.copy(self.__shape_mat)

    @_shape_mat.setter
    def _shape_mat(self, shape_mat: np.ndarray) -> None:
        if shape_mat.ndim != 2:
            throw_error('wrongInput:shape_mat', 'shape_mat must be a matrix')
        if shape_mat.shape[0] != shape_mat.shape[1]:
            throw_error('wrongInput:shape_mat', 'shape_mat must be a square matrix')
        shape_mat = np.array(try_treat_as_real(shape_mat, self._abs_tol), dtype=np.float64)
        if shape_mat.size > 0:
            if not np.all(np.isfinite(shape_mat.flatten())):
                throw_error('wrongInput:shape_mat', 'configuration matrix should contain all finite values')
            if not (is_mat_symm(shape_mat, self._abs_tol) and is_mat_pos_def(shape_mat, self._abs_tol, True)):
                throw_error('wrongInput:shape_mat',
                            'configuration matrix should be symmetric and positive semidefinite')
        self.__shape_mat = np.copy(shape_mat)

    def quad_func(self) -> np.ndarray:
        pass

    def get_shape_mat(self) -> np.ndarray:
        return np.copy(self._shape_mat)

    def _get_scalar_polar_internal(self, is_robust_method: bool):
        pass

    @classmethod
    def from_rep_mat(cls, *args, **kwargs) -> np.ndarray:
        if len(args) == 0:
            throw_error('wrongInput', 'At least one input argument is expected')
        shape_vec = np.array(args[-1]).flatten()
        args = args[:-1]
        ell_obj = Ellipsoid(*args, **kwargs)
        return ell_obj.rep_mat(shape_vec)

    @classmethod
    def from_dict(cls, dict_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        def dict_2_ell(ell_dict: Dict[str, Any]):
            ell_dict = ell_dict.copy()
            shape_mat = ell_dict['shape_mat']
            center_vec = ell_dict['center_vec']
            del ell_dict['shape_mat']
            del ell_dict['center_vec']
            return Ellipsoid(center_vec, shape_mat, **ell_dict)
        dict_arr = np.array(dict_arr)
        return np.reshape(np.array([dict_2_ell(ell_dict) for ell_dict in list(dict_arr.flatten())]), dict_arr.shape)

    @staticmethod
    def _regularize(q_mat: np.ndarray, reg_tol: float) -> np.ndarray:
        from ellipy.gras.la.la import reg_pos_def_mat
        return reg_pos_def_mat(q_mat, reg_tol)

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
                           p_univ_vec: np.ndarray, is_good_dir_vec: np.ndarray) -> Tuple[np.ndarray, bool]:
        __ABS_TOL = 1e-14
        abs_tol = Properties.get_abs_tol()
        _, min_ell_pts_mat = first_ell.rho(first_ell, l_mat)
        _, sub_ell_pts_mat = sec_ell.rho(sec_ell, l_mat)
        if first_ell.dimension(first_ell) == 3:
            is_plot_center_3d = True
        else:
            is_plot_center_3d = False

        def calc_diff(is_good: bool, ind: int) -> np.ndarray:
            if is_good:
                diff_bnd_mat = min_ell_pts_mat[:, ind] - sub_ell_pts_mat[:, ind]
            else:
                _, diff_bnd_mat = rho_mat((1 - p_univ_vec[ind]) * sec_ell.get_shape_mat()
                                          + (1 - 1 / p_univ_vec[ind]) * first_ell.get_shape_mat(),
                                          l_mat[:, ind], abs_tol, first_ell.get_center_vec() - sec_ell.get_center_vec())
            if np.all(np.abs(diff_bnd_mat - first_ell.get_center_vec() +
                             sec_ell.get_center_vec()) < __ABS_TOL):
                diff_bnd_mat = first_ell.get_center_vec() - sec_ell.get_center_vec()
            else:
                nonlocal is_plot_center_3d
                is_plot_center_3d = False
            return diff_bnd_mat

        diff_bound_mat = np.array([calc_diff(x, y) for x, y in zip(is_good_dir_vec, np.arange(0, l_mat.shape[1]))])
        return diff_bound_mat, is_plot_center_3d

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
    def _check_is_me(cls, ell_arr: Union[Iterable, np.ndarray], *args, **kwargs):
        cls._check_is_me_internal(ell_arr, *args, **kwargs)

    def _shape_single_internal(self, is_mod_scal: bool, mod_mat: np.ndarray):
        pass

    def _projection_single_internal(self, ort_basis_mat: np.ndarray):
        self._shape_mat = ort_basis_mat.T @ self.get_shape_mat() @ ort_basis_mat
        self._center_vec = ort_basis_mat.T @ self.get_center_vec()

    @classmethod
    def _check_is_me_virtual(cls, ell_arr: Union[Iterable, np.ndarray], *args, **kwargs):
        cls._check_is_me(ell_arr, *args, **kwargs)

    def _get_single_copy(self):
        center_vec, shape_mat = self.parameters()
        return self.__class__(center_vec, shape_mat,
                              abs_tol=self._abs_tol,
                              rel_tol=self._rel_tol,
                              n_plot_2d_points=self._n_plot_2d_points,
                              n_plot_3d_points=self._n_plot_3d_points)

    def double(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._center_vec, self._shape_mat

    def ellbndr_2d(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def ellbndr_3d(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @classmethod
    def ge(cls, first_ell_arr: Union[Iterable, np.ndarray],
           second_ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def __ge__(self, other):
        return self.ge([self], [other]).flatten()[0]

    @classmethod
    def gt(cls, first_ell_arr: Union[Iterable, np.ndarray],
           second_ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def __gt__(self, other):
        return self.gt([self], [other]).flatten()[0]

    @classmethod
    def le(cls, first_ell_arr: Union[Iterable, np.ndarray],
           second_ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def __le__(self, other):
        return self.le([self], [other]).flatten()[0]

    @classmethod
    def lt(cls, first_ell_arr: Union[Iterable, np.ndarray],
           second_ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def __lt__(self, other):
        return self.lt([self], [other]).flatten()[0]

    @classmethod
    def ne(cls, first_ell_arr: Union[Iterable, np.ndarray],
           second_ell_arr: Union[Iterable, np.ndarray]) -> Tuple[np.ndarray, str]:
        pass

    def __ne__(self, other):
        is_ne, _ = self.ne([self], [other])
        return np.array(is_ne).flatten()[0]

    def get_boundary(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_boundary_by_factor(self, factor_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @classmethod
    def get_inv(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def get_move_2_origin(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def get_projection(cls, ell_arr: Union[Iterable, np.ndarray], basis_mat: np.ndarray) -> np.ndarray:
        proj_ell_arr = cls.get_copy(ell_arr)
        return cls.projection(proj_ell_arr, basis_mat)

    def get_rho_boundary(self, n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def get_rho_boundary_by_factor(self, factor_vec: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @classmethod
    def inv(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def is_bigger(self, sec_ell) -> bool:
        from ellipy.elltool.core.core import ell_sim_diag
        self._check_is_me(sec_ell, 'second')
        self._check_if_scalar(sec_ell)
        sec_ell = np.array(sec_ell).flatten()[0]
        n_dim_vec, n_rank_vec = self.dimension([self, sec_ell], return_rank=True)
        if n_dim_vec[0] != n_dim_vec[1]:
            throw_error('wrongInput', 'both arguments must be single ellipsoids of the same dimension.')
        if n_rank_vec[0] < n_rank_vec[1]:
            return False
        first_shape_mat = self.get_shape_mat()
        sec_shape_mat = sec_ell.get_shape_mat()
        if self.is_degenerate([self]).flatten()[0]:
            if Properties.get_is_verbose():
                logger = get_logger()
                logger.info('IS_BIGGER: Warning! First ellipsoid is degenerate.')
                logger.info('           Regularizing...')
            first_shape_mat = self._regularize(first_shape_mat, self._abs_tol)
        _, abs_tol = self.get_abs_tol([self, sec_ell], lambda z: np.min(z))
        t_mat = ell_sim_diag(first_shape_mat, sec_shape_mat, abs_tol)
        return np.max(np.abs(np.diag(t_mat @ sec_shape_mat @ t_mat.T))) < (1 + self._abs_tol)

    @classmethod
    def is_internal(cls, ell_arr: Union[Iterable, np.ndarray],
                    mat_of_vec_mat: np.ndarray, mode: str) -> np.ndarray:
        pass

    @classmethod
    def minksum_ea(cls, ell_arr: Union[Iterable, np.ndarray], dir_mat: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def minksum_ia(cls, ell_arr: Union[Iterable, np.ndarray], dir_mat: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def move_2_origin(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.double()

    @classmethod
    def polar(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def rho(cls, ell_arr: Union[Iterable, np.ndarray], dirs_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        def f_rho_for_dir(ell_obj, dir_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            c_vec, e_mat = ell_obj.double()
            a_tol = ell_obj.get_abs_tol(ell_obj)
            sup_fun, x_vec = rho_mat(e_mat, dir_vec, a_tol, c_vec.reshape(-1, 1))
            return sup_fun, x_vec

        cls._check_is_me(ell_arr, 'first')
        ell_arr = np.array(ell_arr)
        if not is_numeric(dirs_arr):
            throw_error('wrongInput', 'second argument must be matrix of direction vectors')
        dir_size_vec = dirs_arr.shape
        ell_size_vec = ell_arr.shape
        is_one_ell = ell_arr.size == 1
        is_one_dir = dir_size_vec[1] == 1 and len(dir_size_vec) == 2
        #
        n_ell = np.prod(ell_size_vec)
        n_dim = dir_size_vec[0]
        n_dirs = np.prod(dir_size_vec[1:])
        #
        if not (is_one_ell or is_one_dir or n_ell == n_dirs and
                (ell_size_vec[0] == 1 or ell_size_vec[1] == 1) and
                len(dir_size_vec) == 2 or np.all(ell_size_vec == dir_size_vec[1:])):
            throw_error('wrongInput:wrongSizes', 'arguments must be single ellipsoid or single ' +
                        'direction vector or arrays of almost the same sizes')
        #
        n_dims_arr = cls.dimension(ell_arr)
        if not np.all(n_dims_arr.flatten() == n_dim):
            throw_error('wrongInput', 'dimensions mismatch')
        #
        if is_one_ell:  # one ellipsoid, multiple directions
            cen_vec = ell_arr.flat[0].get_center_vec()
            ell_mat = ell_arr.flat[0].get_shape_mat()
            _, abs_tol = cls.get_abs_tol(ell_arr)
            dirs_mat = np.reshape(dirs_arr, (n_dim, n_dirs))
            #
            sup_arr, bp_arr = rho_mat(ell_mat, dirs_mat, abs_tol, cen_vec.reshape(-1, 1))
            if len(dir_size_vec) > 2:
                sup_arr = np.reshape(sup_arr, dir_size_vec[1:])
                bp_arr = np.reshape(bp_arr, dir_size_vec)
        elif is_one_dir:  # multiple ellipsoids, one direction
            res_c_arr, x_c_arr = zip(*map(lambda ell_obj: f_rho_for_dir(ell_obj, dirs_arr),
                                          ell_arr.flatten()))
            sup_arr = np.array(res_c_arr)
            x_c_arr = np.array(x_c_arr)
            bp_arr = np.hstack(x_c_arr)
            sup_arr = np.reshape(sup_arr, ell_size_vec)
            if len(ell_size_vec) > 2:
                bp_arr = np.reshape(bp_arr, (n_dim, ) + ell_size_vec)
        else:  # multiple ellipsoids, multiple directions
            dir_c_arr = np.reshape(dirs_arr, (n_dim, n_dirs)).T
            #
            res_c_arr, x_c_arr = zip(*map(lambda ell_obj, l_vec:
                                          f_rho_for_dir(ell_obj, l_vec.reshape(-1, 1)),
                                          ell_arr.flatten(), dir_c_arr))
            sup_arr = np.array(res_c_arr)
            x_c_arr = np.array(x_c_arr)
            bp_arr = np.hstack(x_c_arr)
            if len(dir_size_vec) > 2:
                sup_arr = np.reshape(sup_arr, dir_size_vec[1:])
                bp_arr = np.reshape(bp_arr, dir_size_vec)
        return sup_arr, bp_arr

    @classmethod
    def to_dict(cls, ell_arr: Union[Iterable, np.ndarray],
                is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[np.ndarray, Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.ndarray], np.ndarray]]]:
        def ell_2_dict(ell_obj, is_prop_incl: bool) -> dict:
            ell_center_vec, ell_shape_mat = ell_obj.parameters()
            ell_dict = {
                'shape_mat': ell_shape_mat,
                'center_vec': ell_center_vec
            }
            if is_prop_incl:
                ell_obj_arr = np.array([ell_obj])
                ell_dict['abs_tol'] = ell_obj.get_abs_tol(ell_obj_arr, None).flatten()[0]
                ell_dict['rel_tol'] = ell_obj.get_rel_tol(ell_obj_arr, None).flatten()[0]
                ell_dict['n_plot_2d_points'] = \
                    ell_obj.get_property(ell_obj_arr, 'n_plot_2d_points', None).flatten()[0]
                ell_dict['n_plot_3d_points'] = \
                    ell_obj.get_property(ell_obj_arr, 'n_plot_3d_points', None).flatten()[0]
            return ell_dict

        ell_arr = np.array(ell_arr)
        if abs_tol is None:
            _, abs_tol = cls.get_abs_tol(ell_arr)
        ell_dict_arr = np.reshape(np.array(
            [ell_2_dict(ell_obj, is_prop_included) for ell_obj in list(ell_arr.flatten())]), ell_arr.shape)
        field_nice_names_dict = {
            'shape_mat': 'QSqrt',
            'center_vec': 'q'
        }
        field_transform_func_dict = {
            'shape_mat': lambda z: sqrtm_pos(z),
            'center_vec': lambda z: z
        }
        field_descr_dict = {
            'shape_mat': 'Ellipsoid shape matrix.',
            'center_vec': 'Ellipsoid center vector.'
        }
        if is_prop_included:
            field_nice_names_dict['abs_tol'] = 'abs_tol'
            field_descr_dict['abs_tol'] = 'Absolute tolerance.'
            field_nice_names_dict['rel_tol'] = 'rel_tol'
            field_descr_dict['rel_tol'] = 'Relative tolerance.'
            field_nice_names_dict['n_plot_2d_points'] = 'n_plot_2d_points'
            field_descr_dict['n_plot_2d_points'] = 'Degree of ellipsoid border smoothness in 2D plotting.'
            field_nice_names_dict['n_plot_3d_points'] = 'n_plot_3d_points'
            field_descr_dict['n_plot_3d_points'] = 'Degree of ellipsoid border smoothness in 3D plotting.'
        return ell_dict_arr, field_nice_names_dict, field_descr_dict, field_transform_func_dict

    @classmethod
    def uminus(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def __neg__(self):
        return self.uminus([self]).flatten()[0]

    def __str__(self):
        res_str = ['\n']
        ell_dict, field_names_dict, field_descr_dict, _ = self.to_dict([self], False)
        ell_dict = np.array(ell_dict).flatten()[0]

        prop_dict = {'actualClass': 'Ellipsoid', 'shape': '()'}
        res_str.append('-------ellipsoid object-------\n')
        res_str.append('Properties:\n')
        res_str.append(str(prop_dict))
        res_str.append('\n')
        res_str.append('Fields (name, type, description):\n')
        res_str.append('    {}    float64    {}\n'.format(
            field_names_dict['shape_mat'], field_descr_dict['shape_mat']))
        res_str.append('    {}    float64    {}\n'.format(
            field_names_dict['center_vec'], field_descr_dict['center_vec']))
        res_str.append('\nData: \n')
        res_str.append(str(ell_dict))
        return ''.join(res_str)
