from ellipy.elltool.core.aellipsoid.AEllipsoid import *
from ellipy.elltool.core.abasicellipsoid.ABasicEllipsoid import *
from ellipy.elltool.conf.properties.Properties import Properties
from ellipy.gras.la.la import is_mat_pos_def, is_mat_symm, try_treat_as_real, sqrtm_pos
from ellipy.gen.common.common import throw_error
from ellipy.gen.logging.logging import get_logger
from ellipy.gras.geom.tri.tri import sphere_tri_ext
from ellipy.gras.geom.ell.ell import rho_mat
from typing import Union, Tuple, Dict, Callable, Any
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

    def get_center_vec(self) ->np.ndarray:
        return np.copy(self._center_vec)

    def _get_scalar_polar_internal(self, is_robust_method: bool = True):
        from ellipy.gras.geom.ell.ell import quad_mat
        self._check_if_scalar(self)
        if not isinstance(is_robust_method, bool):
            throw_error('wrongInput', 'is_robust_method should be bool')

        if is_robust_method:
            c_vec, sh_mat = self.double()
            inv_sh_mat = np.linalg.inv(sh_mat)
            inv_sh_mat = (inv_sh_mat + inv_sh_mat.T)/2

            norm_const = quad_mat(sh_mat, c_vec, 0, 'inv')
            polar_c_vec = -np.linalg.lstsq(sh_mat, c_vec, -1)[0] / (1.0 - norm_const)
            polar_sh_mat = inv_sh_mat / (1.0 - norm_const) + polar_c_vec * polar_c_vec.T
            polar_obj = self.__class__(polar_c_vec, polar_sh_mat)
            return polar_obj
        else:
            c_vec, sh_mat = self.double()
            is_zero_in_ell = quad_mat(sh_mat, c_vec, 0, 'invadv')
            if is_zero_in_ell < 1:
                aux_mat = np.linalg.inv(sh_mat - c_vec @ c_vec.T)
                aux_mat = (aux_mat + aux_mat.T) / 2

                d_vec, v_mat = np.linalg.eigh(aux_mat)
                m_mat = v_mat @ np.diag(d_vec) @ v_mat.T
                aux_mat = 0.5 * (m_mat + m_mat.T)
                polar_c_vec = -aux_mat @ c_vec
                polar_sh_mat = (1 + quad_mat(aux_mat, c_vec, 0, 'plain')) * aux_mat
                polar_obj = self.__class__(polar_c_vec, polar_sh_mat)
                return polar_obj
            else:
                throw_error('degenerateEllipsoid', 'The resulting ellipsoid is not bounded')

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

    def _get_grid_by_factor(self, factor_vec: np.ndarray = np.array(1., dtype=np.float64)):
        __EPS = 1e-15
        n_dim = self.dimension([self]).flat[0]

        if n_dim < 2 or n_dim > 3:
            throw_error('wrongDim:ellipsoid', 'ellipsoid must be of dimension 2 or 3')

        if factor_vec.ndim == 0:
            factor = factor_vec.flat[0]
        else:
            factor = factor_vec.flat[n_dim - 2]

        if n_dim == 2:
            n_plot_points = self._n_plot_2d_points
            if not (factor == 1):
                n_plot_points = np.floor(n_plot_points * factor)
        else:
            n_plot_points = self._n_plot_3d_points
            if not (factor == 1):
                n_plot_points = np.floor(n_plot_points * factor)
        v_grid_mat, f_grid_mat = sphere_tri_ext(n_dim, n_plot_points)
        v_grid_mat[v_grid_mat == 0] = __EPS

        return v_grid_mat, f_grid_mat

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

    def get_boundary(self, n_points: int = None, return_grid: bool = False) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        Ellipsoid._check_if_scalar(self)
        n_dim = self.dimension([self]).flat[0]
        if n_dim == 2:
            if n_points is None:
                n_points = self._n_plot_2d_points
        elif n_dim == 3:
            if n_points is None:
                n_points = self._n_plot_3d_points
        else:
            throw_error('wrongDim', 'ellipsoid must be of dimension 2 or 3')

        if return_grid:
            dir_mat, f_mat = sphere_tri_ext(n_dim, n_points, return_grid)
        else:
            dir_mat = sphere_tri_ext(n_dim, n_points, return_grid)
            f_mat = None
        cen_vec, q_mat = self.double()
        ret_mat = dir_mat @ sqrtm_pos(q_mat, self.get_abs_tol([self], f_prop_fun=None).flat[0])
        cen_mat = np.tile(cen_vec.T, (dir_mat.shape[0], 1))
        ret_mat = ret_mat + cen_mat
        if return_grid:
            return ret_mat, f_mat
        else:
            return ret_mat

    def get_boundary_by_factor(self, factor_vec: Union[np.ndarray, float] = None, return_grid: bool = False) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        Ellipsoid._check_if_scalar(self)
        n_dim = self.dimension([self]).flat[0]
        if (n_dim < 2) or (n_dim > 3):
            throw_error('wrongDim', 'ellipsoid must be of dimension 2 or 3')
        if factor_vec is None:
            factor = 1
        elif type(factor_vec) is int:
            factor = factor_vec
        else:
            factor = factor_vec[n_dim - 2]
        factor = np.float64(factor)
        if n_dim == 2:
            n_plot_points = self._n_plot_2d_points
            if factor != 1.0:
                n_plot_points = int(np.floor(n_plot_points * factor))
        else:
            n_plot_points = self._n_plot_3d_points
            if factor != 1.0:
                n_plot_points = int(np.floor(n_plot_points * factor))
        return self.get_boundary(n_plot_points, return_grid)

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

    def get_rho_boundary(self, n_points: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Ellipsoid._check_if_scalar(self)
        n_dim = self.dimension([self]).flat[0]
        if n_dim == 2:
            if n_points is None:
                n_points = self._n_plot_2d_points
        elif n_dim == 3:
            if n_points is None:
                n_points = self._n_plot_3d_points
        else:
            throw_error('wrongDim', 'ellipsoid must be of dimension 2 or 3')
        dir_mat, f_mat = sphere_tri_ext(n_dim, n_points, True)
        cen_vec, q_mat = self.double()
        l_grid_mat = np.vstack((dir_mat, dir_mat[0, :]))
        abs_tol = self.get_abs_tol([self], f_prop_fun=None).flat[0]
        sup_vec, bp_mat = rho_mat(q_mat, l_grid_mat.T, abs_tol, np.expand_dims(cen_vec, 1))
        sup_vec = sup_vec.T
        bp_mat = bp_mat.T
        return bp_mat, f_mat, sup_vec, l_grid_mat

    def get_rho_boundary_by_factor(self, factor_vec: Union[np.ndarray, float] = None) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Ellipsoid._check_if_scalar(self)
        n_dim = self.dimension([self]).flat[0]
        n_plot_points = None
        if factor_vec is None:
            factor = 1
        elif type(factor_vec) is int:
            factor = factor_vec
        else:
            factor = factor_vec[n_dim - 2]
        factor = np.float64(factor)
        if n_dim == 2:
            n_plot_points = self._n_plot_2d_points
            if factor != 1.0:
                n_plot_points = int(np.floor(n_plot_points * factor))
        elif n_dim == 3:
            n_plot_points = self._n_plot_3d_points
            if factor != 1.0:
                n_plot_points = int(np.floor(n_plot_points * factor))
        else:
            throw_error('wrongDim', 'ellipsoid must be of dimension 2 or 3')
        return self.get_rho_boundary(n_points=n_plot_points)

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
        def is_int_single_vec(ell_arr_in, x_vec, mode_in) -> bool:
            def f_single_case(sing_ell: np.ndarray, abs_tol: np.float64) -> bool:
                from ellipy.gras.geom.ell.ell import inv_mat
                is_pos = False
                c_vec = (x_vec - np.array([sing_ell.flat[0].get_center_vec().T])).T
                sh_mat = sing_ell.flat[0].get_shape_mat()

                if sing_ell.flat[0].is_degenerate([sing_ell]).flat[0]:
                    if Properties.get_is_verbose():
                        logger = get_logger()
                        logger.info('ISINTERNAL: Warning! There is degenerate ellipsoid in the array.')
                        logger.info('           Regularizing...')
                    sh_mat = Ellipsoid._regularize(sh_mat, abs_tol)
                r_val = np.sqrt(c_vec.T @ inv_mat(sh_mat) @ c_vec)
                if r_val < 1.0 or np.all(np.abs(r_val - 1.0) < abs_tol):
                    is_pos = True
                return is_pos

            abs_tol_arr = cls.get_abs_tol(ell_arr_in, f_prop_fun=None)
            is_pos_arr = np.full((n_dims_mat.size, 1), True, dtype=bool)
            for j in range(0, n_dims_mat.size):
                is_pos_arr[j] = f_single_case(ell_arr_in[j], abs_tol_arr[j])
            if mode_in == 'u':
                is_positive = False
                if np.any(np.ravel(is_pos_arr)):
                    is_positive = True
            else:
                is_positive = True
                if not (np.all(np.ravel(is_pos_arr))):
                    is_positive = False
            return is_positive

        cls._check_is_me(ell_arr, 'first')
        if not is_numeric(mat_of_vec_mat):
            throw_error('wrongInput', 'The second input argument must be a numeric matrix.')
        if ell_arr.size == 0:
            throw_error('wrongInput:emptyArray', 'Each array must be not empty.')
        if np.any(cls.is_empty(np.ravel(ell_arr))):
            throw_error('wrongInput:emptyEllipsoid', 'Array should not have empty ellipsoid.')

        if not isinstance(mode, str):
            mode = 'u'
        if not (mode == 'u' or mode == 'i'):
            throw_error('wrongInput', "third argument is expected to be either 'u', or 'i'.")

        n_dims_mat = cls.dimension(ell_arr)
        m_rows, m_cols = mat_of_vec_mat.shape
        if not np.all(np.ravel(n_dims_mat) == m_rows):
            throw_error('wrongSizes', 'dimensions mismatch.')

        l_c_vec = np.reshape(mat_of_vec_mat, (m_rows, m_cols))
        is_positive_vec = np.full((np.size(l_c_vec, 1), 1), True, dtype=bool)
        for i in range(0, np.size(l_c_vec, 1)):
            is_positive_vec[i] = is_int_single_vec(ell_arr, l_c_vec[:, i], mode)

        return is_positive_vec

    @classmethod
    def minksum_ea(cls, ell_arr: Union[Iterable, np.ndarray], dir_mat: np.ndarray) -> np.array:
        def f_single_direction(dir_ind: int) -> Ellipsoid:
            def f_add_sh(sing_ell: Ellipsoid, abs_tol: float) -> Tuple[np.array, float]:
                from ellipy.gras.geom.ell.ell import quad_mat
                sh_mat = sing_ell.get_shape_mat()
                if sing_ell.is_degenerate([sing_ell]).flat[0]:
                    if Properties.get_is_verbose():
                        logger = get_logger()
                        logger.info('MINKSUM_IA: Warning! Degenerate ellipsoid.')
                        logger.info('           Regularizing...')
                    sh_mat = cls._regularize(sh_mat, abs_tol)
                fst_coef = np.sqrt(quad_mat(sh_mat, dir_vec))
                return ((1 / fst_coef) * sh_mat), fst_coef

            if dir_mat.ndim > 1:
                dir_vec = dir_mat[:, dir_ind]
            else:
                dir_vec = dir_mat
            sub_sh_mat, sec_coef = zip(*list(map(lambda x, y: f_add_sh(x, y), ell_arr, abs_tol_arr)))
            sub_sh_mat = np.sum(np.array(sub_sh_mat), 0)
            sec_coef = np.sum(np.array(sec_coef))
            sub_sh_mat = 0.5 * sec_coef * (sub_sh_mat + sub_sh_mat.T)
            return ell_arr[0].__class__(cent_vec, sub_sh_mat)

        ell_arr = np.array(ell_arr).flatten()
        cls._check_is_me(ell_arr, 'first')
        n_ell = ell_arr.size
        n_dims_ell_arr = cls.dimension(ell_arr)
        n_dims = dir_mat.shape[0]
        if dir_mat.ndim <= 1:
            n_cols = 1
        else:
            n_cols = dir_mat.shape[1]
        if n_ell == 0:
            throw_error('wrongInput:emptyArray', 'Each array must be not empty.')
        if np.any(ell_arr[0].is_empty(ell_arr)):
            throw_error('wrongInput:emptyEllipsoid', 'Array should not have empty ellipsoid.')
        if not ((np.all(n_dims_ell_arr == n_dims)) and (np.all(n_dims_ell_arr == n_dims_ell_arr[0]))):
            throw_error('wrongSizes:', 'ellipsoids in the array and vector(s) must be of the same dimension.')

        if n_ell == 1:
            return np.squeeze(ell_arr[0].rep_mat([1, n_cols]))
        else:
            cent_vec = np.sum(np.array([ell.get_center_vec() for ell in ell_arr]), 0)
            abs_tol_arr = cls.get_abs_tol(ell_arr, f_prop_fun=None).flatten()
            return np.array([f_single_direction(i) for i in np.arange(n_cols)])

    @classmethod
    def minksum_ia(cls, ell_arr: Union[Iterable, np.ndarray], dir_mat: np.ndarray) -> np.ndarray:
        def f_rot_arr(ell_ind: int) -> Tuple[np.ndarray, np.ndarray]:
            from ellipy.gras.la.la import ml_orth_transl
            ell_obj = ell_arr[ell_ind]
            abs_tol = abs_tol_arr[ell_ind]
            sh_mat = ell_obj.get_shape_mat()
            if ell_obj.is_degenerate([ell_obj]).flat[0]:
                if Properties.get_is_verbose():
                    logger = get_logger()
                    logger.info('MINKSUM_IA: Warning! Degenerate ellipsoid.')
                    logger.info('           Regularizing...')
                sh_mat = cls._regularize(sh_mat, abs_tol)
            sh_sqrt_mat = sqrtm_pos(sh_mat, abs_tol)
            dst_mat = sh_sqrt_mat @ dir_mat
            return sh_sqrt_mat, np.reshape(ml_orth_transl(dst_mat, src_mat), [n_dims, n_dims, n_cols])

        def f_single_direction(dir_ind: int) -> Ellipsoid:
            sub_sh_mat = np.cumsum(np.array(list(map(lambda ell_ind:
                                                     rot_arr[ell_ind, :, :, dir_ind]
                                                     @ sqrt_sh_arr[ell_ind, :, :],
                                                     np.arange(0, n_ell)))), 0)
            sub_sh_mat = sub_sh_mat[-1, :, :]
            sh_mat = sub_sh_mat.T @ sub_sh_mat
            return ell_arr[0].__class__(cent_vec, sh_mat)
        #
        ell_arr = np.array(ell_arr).flatten()
        cls._check_is_me(ell_arr, 'first')
        n_ell = ell_arr.size
        n_dims_ell_arr = cls.dimension(ell_arr)
        n_dims = dir_mat.shape[0]
        if dir_mat.ndim <= 1:
            n_cols = 1
        else:
            n_cols = dir_mat.shape[1]

        if not n_ell > 0:
            throw_error('wrongInput:emptyArray', 'Each array must be not empty.')
        if np.any(ell_arr[0].is_empty(ell_arr)):
            throw_error('wrongInput:emptyEllipsoid', 'Array should not have empty ellipsoid.')
        if not ((np.all(n_dims_ell_arr == n_dims)) and (np.all(n_dims_ell_arr == n_dims_ell_arr[0]))):
            throw_error('wrongSizes:', 'ellipsoids in the array and vector(s) must be of the same dimension.')
        if n_ell == 1:
            return np.squeeze(ell_arr[0].rep_mat([1, n_cols]))
        else:
            cent_vec = np.sum(np.array([ell.get_center_vec() for ell in ell_arr]), 0)
            abs_tol_arr = cls.get_abs_tol(ell_arr, f_prop_fun=None).flatten()

            src_mat = sqrtm_pos(ell_arr[0].get_shape_mat(), np.min(abs_tol_arr)) @ dir_mat
            sqrt_sh_arr, rot_arr = zip(*[f_rot_arr(i) for i in np.arange(0, n_ell)])
            sqrt_sh_arr = np.array(sqrt_sh_arr)
            rot_arr = np.array(rot_arr)
            return np.array([f_single_direction(i) for i in np.arange(0, n_cols)])

    @classmethod
    def move_2_origin(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    def parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.double()

    @classmethod
    def polar(cls, ell_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        cls._check_is_me(ell_arr)
        if np.any(cls.is_degenerate(ell_arr)):
            throw_error('degenerateEllipsoid', 'The resulting ellipsoid is not bounded')
        pol_ell_arr = np.empty((1, ell_arr.size), dtype=cls.__class__)
        for i_elem in range(ell_arr.size):
            pol_ell_arr[i_elem] = cls._get_scalar_polar_internal(ell_arr[i_elem], True)
        return np.reshape(pol_ell_arr, ell_arr.shape)

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
                bp_arr = np.reshape(bp_arr, (n_dim,) + ell_size_vec)
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
        cls._check_is_me(ell_arr)
        ell_arr = np.array(ell_arr)
        size_vec = np.shape(ell_arr)
        if 0 == ell_arr.size:
            return np.empty(shape=size_vec, dtype=cls.__class__)
        else:
            ell_flat_arr = ell_arr.flatten()

            def f_single_uminus(ell_obj):
                return ell_obj.__class__(-ell_obj.get_center_vec(),
                                         ell_obj.get_shape_mat())

            return np.reshape(np.array(list(map(f_single_uminus, ell_flat_arr))), newshape=size_vec)

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
