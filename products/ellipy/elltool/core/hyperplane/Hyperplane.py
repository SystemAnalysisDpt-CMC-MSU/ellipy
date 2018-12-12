from ellipy.elltool.core.abasicellipsoid.ABasicEllipsoid import *
from ellipy.elltool.conf.properties.Properties import Properties
from ellipy.gras.la.la import try_treat_as_real
from typing import Tuple, Dict, Callable, Any
import numpy as np
import copy


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

    def __init__(self, hyp_norm_vec: np.ndarray = np.array([0.], dtype=np.float64),
                 hyp_shift=0., **kwargs):
        ABasicEllipsoid.__init__(self)
        prop_list, _ = Properties.parse_prop(kwargs, ['abs_tol', 'rel_tol'])
        self._abs_tol = prop_list[0]
        self._rel_tol = prop_list[1]
        self._normal_vec = hyp_norm_vec
        self._shift = hyp_shift

    @property
    def _normal_vec(self) -> np.ndarray:
        return np.copy(self.__normal_vec)

    @_normal_vec.setter
    def _normal_vec(self, normal_vec: np.ndarray):
        if type(normal_vec) != np.ndarray or not is_numeric(normal_vec):
            throw_error('wrongInput:normal_vec', 'normal_vec should be numeric array')
        normal_vec = np.array(try_treat_as_real(normal_vec), dtype=np.float64)
        if normal_vec.size != np.max(normal_vec.shape):
            throw_error('wrongInput:normal_vec', 'normal_vec should be a vector')
        normal_vec = normal_vec.flatten()
        if not np.all(np.isfinite(normal_vec)):
            throw_error('wrongInput:normal_vec', 'normal_vec should have all finite values')
        self.__normal_vec = np.copy(normal_vec)

    @property
    def _shift(self) -> np.float64:
        return self.__shift

    @_shift.setter
    def _shift(self, shift: np.float64):
        if not is_numeric(shift):
            throw_error('wrongInput:shift', 'shift should be a number')
        shift = np.array(try_treat_as_real(shift), dtype=np.float64).flatten()
        if shift.size != 1:
            throw_error('wrongInput:shift', 'shift should be a number')
        shift = shift[0]
        if not np.isfinite(shift):
            throw_error('wrongInput:shift', 'shift should have a finite value')
        self.__shift = shift

    @classmethod
    def from_rep_mat(cls, *args, **kwargs) -> np.ndarray:
        if len(args) == 0:
            throw_error('wrongInput', 'At least one input argument is expected')
        shape_vec = np.array(args[-1]).flatten()
        args = args[:-1]
        hp_obj = Hyperplane(*args, **kwargs)
        return hp_obj.rep_mat(shape_vec)

    @classmethod
    def from_dict(cls, dict_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        def dict_2_hp(hyp_dict: Dict[str, Any]):
            hyp_dict = hyp_dict.copy()
            normal_vec = hyp_dict['normal_vec']
            shift = hyp_dict['shift']
            del hyp_dict['normal_vec']
            del hyp_dict['shift']
            return Hyperplane(normal_vec, shift, **hyp_dict)
        dict_arr = np.array(dict_arr)
        return np.reshape(np.array([dict_2_hp(hp_dict) for hp_dict in list(dict_arr.flatten())]), dict_arr.shape)

    @classmethod
    def _check_is_me(cls, hyp_arr: Union[Iterable, np.ndarray], *args, **kwargs):
        cls._check_is_me_internal(hyp_arr, *args, **kwargs)

    @classmethod
    def _check_is_me_virtual(cls, hyp_arr: Union[Iterable, np.ndarray], *args, **kwargs):
        cls._check_is_me(hyp_arr, *args, **kwargs)

    def _get_single_copy(self):
        normal_vec, shift = self.parameters()
        return self.__class__(normal_vec, shift, abs_tol=self._abs_tol)

    @classmethod
    def contains(cls, hyp_arr: Union[Iterable, np.ndarray], x_arr: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def dimension(cls, hyp_arr: Union[Iterable, np.ndarray], return_rank=False) -> np.ndarray:
        cls._check_is_me(hyp_arr)
        hyp_arr = np.array(hyp_arr)

        def sing_dimension(hyp_obj) -> int:
            cur_abs_tol = hyp_obj.get_abs_tol(np.array([hyp_obj]), None).flatten()[0]
            normal_vec, shift = hyp_obj.parameters()
            normal_vec = normal_vec.flatten()
            n_sub_dim = normal_vec.size
            if n_sub_dim < 2:
                if np.abs(normal_vec[0]) <= cur_abs_tol and np.abs(shift) <= cur_abs_tol:
                    n_dim = 0
                else:
                    n_dim = n_sub_dim
            else:
                n_dim = n_sub_dim
            return n_dim

        return np.reshape(
            np.array([sing_dimension(hyp_obj) for hyp_obj in list(hyp_arr.flatten())]), hyp_arr.shape)

    def double(self) -> Tuple[np.ndarray, np.float64]:
        return self._normal_vec, self._shift

    def parameters(self) -> Tuple[np.ndarray, np.float64]:
        return self.double()

    @classmethod
    def is_parallel(cls, first_hyp_arr: Union[Iterable, np.ndarray],
                    sec_hyp_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def ne(cls, first_hyp_arr: Union[Iterable, np.ndarray],
           second_hyp_arr: Union[Iterable, np.ndarray]) -> Tuple[np.ndarray, str]:
        pass

    def __ne__(self, other):
        is_ne, _ = self.ne([self], [other])
        return np.array(is_ne).flatten()[0]

    @classmethod
    def to_dict(cls, hyp_arr: Union[Iterable, np.ndarray],
                is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[np.ndarray, Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.ndarray], np.ndarray]]]:
        def hp_2_dict(hp_obj, is_prop_incl: bool) -> dict:
            if hp_obj.is_empty(np.array([hp_obj])).flatten()[0]:
                hp_dict = {
                    'normal_vec': np.zeros((0,), dtype=np.float64),
                    'shift': np.zeros((0,), dtype=np.float64)
                }
            else:
                hp_norm_vec, hp_scal = hp_obj.parameters()
                hp_norm_vec = np.array(hp_norm_vec).flatten()
                norm_mult = 1. / np.linalg.norm(hp_norm_vec)
                hp_norm_vec = hp_norm_vec * norm_mult
                hp_scal = hp_scal * norm_mult
                if hp_scal < 0.:
                    hp_scal = -hp_scal
                    hp_norm_vec = -hp_norm_vec
                hp_dict = {
                    'normal_vec': hp_norm_vec,
                    'shift': hp_scal
                }
            if is_prop_incl:
                hp_obj_arr = np.array([hp_obj])
                hp_dict['abs_tol'] = hp_obj.get_abs_tol(hp_obj_arr, None).flatten()[0]
                hp_dict['rel_tol'] = hp_obj.get_rel_tol(hp_obj_arr, None).flatten()[0]
            return hp_dict

        hyp_arr = np.array(hyp_arr)
        hyp_dict_arr = np.reshape(np.array(
            [hp_2_dict(hp_obj, is_prop_included) for hp_obj in list(hyp_arr.flatten())]), hyp_arr.shape)
        field_nice_names_dict = {
            'normal_vec': 'normal',
            'shift': 'shift'
        }
        field_transform_func_dict = {
            'normal_vec': lambda z: z,
            'shift': lambda z: z
        }
        field_descr_dict = {
            'normal_vec': 'Hyperplane normal.',
            'shift': 'Hyperplane shift along normal from origin.'
        }
        if is_prop_included:
            field_nice_names_dict['abs_tol'] = 'abs_tol'
            field_descr_dict['abs_tol'] = 'Absolute tolerance.'
            field_nice_names_dict['rel_tol'] = 'rel_tol'
            field_descr_dict['rel_tol'] = 'Relative tolerance.'
        return hyp_dict_arr, field_nice_names_dict, field_descr_dict, field_transform_func_dict

    @classmethod
    def uminus(cls, hyp_arr: Union[Iterable, np.ndarray]) -> np.ndarray:
        cls._check_is_me(hyp_arr)
        size_vec = np.shape(hyp_arr)
        n_elems = np.size(hyp_arr)
        out_hyp_arr = copy.deepcopy(hyp_arr)

        def set_prop(i_obj=None):
            if i_obj is None:
                out_hyp_arr._normal_vec = -out_hyp_arr._normal_vec
                out_hyp_arr._shift = -out_hyp_arr._shift
            else:
                out_hyp_arr[i_obj]._normal_vec = -out_hyp_arr[i_obj]._normal_vec
                out_hyp_arr[i_obj]._shift = -out_hyp_arr[i_obj]._shift
        if type(hyp_arr) != cls:
            [set_prop(i) for i in range(n_elems)]
            out_hyp_arr = np.reshape(out_hyp_arr, newshape=size_vec)
        else:
            set_prop()

        return out_hyp_arr

    def __neg__(self):
        return self.uminus([self]).flatten()[0]

    def __str__(self):
        res_str = ['\n']
        hyp_dict, field_names_dict, field_descr_dict, _ = self.to_dict([self], False)
        hyp_dict = np.array(hyp_dict).flatten()[0]

        prop_dict = {'actualClass': 'Hyperplane', 'shape': '()'}
        res_str.append('-------hyperplane object-------\n')
        res_str.append('Properties:\n')
        res_str.append(str(prop_dict))
        res_str.append('\n')
        res_str.append('Fields (name, type, description):\n')
        res_str.append('    {}    float64    {}\n'.format(
            field_names_dict['normal_vec'], field_descr_dict['normal_vec']))
        res_str.append('    {}    float64    {}\n'.format(
            field_names_dict['shift'], field_descr_dict['shift']))
        res_str.append('\nData: \n')
        res_str.append(str(hyp_dict))
        return ''.join(res_str)
