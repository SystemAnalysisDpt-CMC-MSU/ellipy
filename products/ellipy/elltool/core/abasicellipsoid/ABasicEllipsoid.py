from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable, Union, List, Optional
from ellipy.gen.common.common import throw_error, is_numeric
from ellipy.gen.dict.dict import dict_compare_vec
from ellipy.gras.la.la import try_treat_as_real
import numpy as np


class ABasicEllipsoid(ABC):
    @classmethod
    def _get_prop_aliases_dict(cls) -> Dict[str, str]:
        return {
            'absTol': '_abs_tol',
            'abs_tol': '_abs_tol',
            '_abs_tol': '_abs_tol',
            'relTol': '_rel_tol',
            'rel_tol': '_rel_tol',
            '_rel_tol': '_rel_tol'
        }

    def _before_get_abs_tol(self) -> None:
        pass

    @property
    def _abs_tol(self) -> float:
        self._before_get_abs_tol()
        return self.__abs_tol

    @_abs_tol.setter
    def _abs_tol(self, abs_tol: float):
        self.__abs_tol = abs_tol

    def _before_get_rel_tol(self) -> None:
        pass

    @property
    def _rel_tol(self) -> float:
        self._before_get_rel_tol()
        return self.__rel_tol

    @_rel_tol.setter
    def _rel_tol(self, rel_tol: float):
        self.__rel_tol = rel_tol

    @classmethod
    @abstractmethod
    def dimension(cls, ell_arr: np.ndarray, return_rank=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        pass

    @classmethod
    @abstractmethod
    def to_dict(cls, ell_arr: np.ndarray, is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[List[dict], Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.ndarray], np.ndarray]]]:
        pass

    @classmethod
    def _check_if_scalar(cls, ell_arr: np.ndarray,
                         err_msg: str = 'input argument must be single ellipsoid') -> None:
        if type(ell_arr) == np.ndarray:
            if ell_arr.size != 1:
                throw_error('wrongInput:ell_arr', err_msg)
            if not isinstance(ell_arr.flatten()[0], cls):
                throw_error('wrongInput:ell_arr', err_msg)
        elif not isinstance(ell_arr, cls):
            throw_error('wrongInput:ell_arr', err_msg)

    @classmethod
    def _check_is_me_internal(cls, ell_arr: np.ndarray,
                              var_name: str = '', err_tag: str = 'wrongInput', err_msg: str = None) -> None:
        if var_name != '':
            err_tag += ':' + var_name
        if err_msg is None:
            err_msg = 'input argument must be {}'.format(cls.__name__)
        if type(ell_arr) == np.ndarray:
            if ell_arr.size > 0:
                ell_flatten_arr = ell_arr.flatten()
                for i_ell in range(ell_flatten_arr.size):
                    if not isinstance(ell_flatten_arr[i_ell], cls):
                        throw_error(err_tag, err_msg)
        elif not isinstance(ell_arr, cls):
            throw_error(err_tag, err_msg)

    @classmethod
    @abstractmethod
    def _check_is_me_virtual(cls, ell_arr: np.ndarray, *args, **kwargs):
        pass

    @abstractmethod
    def _get_single_copy(self):
        pass

    @classmethod
    def get_property(cls, ell_arr: np.ndarray, prop_name: str,
                     f_prop_fun: Optional[Callable[[np.ndarray], np.float64]] = lambda x: np.min(x)) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.float64]]:
        cls._check_is_me_internal(ell_arr)
        ell_arr = np.array(ell_arr).flatten()
        ell_flatten_arr = ell_arr.flatten()
        n_elems = ell_flatten_arr.size
        if n_elems == 0:
            prop_arr = np.zeros(ell_arr.shape, dtype=np.float64)
            if f_prop_fun is None:
                return prop_arr
            else:
                return prop_arr, np.nan
        else:
            prop_name = cls._get_prop_aliases_dict()[prop_name]
            prop_arr = np.zeros((n_elems,), dtype=np.float64)
            for i_elem in range(n_elems):
                prop_arr[i_elem] = ell_flatten_arr[i_elem].__getattribute__(prop_name)
            if f_prop_fun is None:
                prop_val = None
            else:
                prop_val = f_prop_fun(prop_arr)
            prop_arr = np.reshape(prop_arr, ell_arr.shape)
            if f_prop_fun is None:
                return prop_arr
            else:
                return prop_arr, prop_val

    @classmethod
    def get_abs_tol(cls, ell_arr: np.ndarray, *args, **kwargs) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.float64]]:
        return cls.get_property(ell_arr, '_abs_tol', *args, **kwargs)

    @classmethod
    def get_rel_tol(cls, ell_arr: np.ndarray, *args, **kwargs) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.float64]]:
        return cls.get_property(ell_arr, '_rel_tol', *args, **kwargs)

    @classmethod
    def _is_equal_internal(cls, ell_first_arr, ell_sec_arr,
                           is_prop_included: bool) -> Tuple[np.ndarray, str]:
        cls._check_is_me_virtual(ell_first_arr, 'first')
        cls._check_is_me_virtual(ell_sec_arr, 'second')
        ell_first_arr = np.array(ell_first_arr)
        ell_sec_arr = np.array(ell_sec_arr)
        if ell_first_arr.size == 0 and ell_sec_arr.size == 0:
            return np.ones((1,), dtype=bool), ''
        elif ell_first_arr.size == 0 or ell_sec_arr.size == 0:
            throw_error('wrongInput', 'input ellipsoidal arrays should be empty at the same time')
        first_shape_vec = ell_first_arr.shape
        sec_shape_vec = ell_sec_arr.shape
        isn_first_scalar = ell_first_arr.size > 1
        isn_sec_scalar = ell_sec_arr.size > 1
        if isn_first_scalar and isn_sec_scalar and first_shape_vec != sec_shape_vec:
            throw_error('wrongSizes', 'sizes of ellipsoidal arrays do not match')

        def get_tol(inp_first_arr, inp_sec_arr):
            inp_arr = np.hstack((inp_first_arr.flatten(), inp_sec_arr.flatten()))
            _, out_abs_tol = cls.get_abs_tol(inp_arr)
            _, out_rel_tol = cls.get_rel_tol(inp_arr)
            return out_abs_tol, out_rel_tol

        abs_tol, rel_tol = get_tol(ell_first_arr, ell_sec_arr)

        ell_first_flat_arr = ell_first_arr.flatten()
        ell_sec_flat_arr = ell_sec_arr.flatten()
        dict_ell1_list, field_nice_names1_dict, _, field_transform_func1_dict = \
            cls.to_dict(ell_first_flat_arr, is_prop_included, abs_tol)
        dict_ell2_list, field_nice_names2_dict, _, field_transform_func2_dict = \
            cls.to_dict(ell_sec_flat_arr, is_prop_included, abs_tol)

        def form_comp_dict(dict_ell: Dict[str, np.array], field_nice_names_dict: Dict[str, str],
                           field_transform_func_dict: Dict[str, Callable[[np.ndarray], np.ndarray]]) -> \
                Dict[str, np.ndarray]:
            comp_dict = dict()
            for field_name in field_nice_names_dict.keys():
                field_nice_name = field_nice_names_dict[field_name]
                field_val_arr = dict_ell[field_name]
                if field_val_arr.size == 0:
                    comp_dict[field_nice_name] = np.zeros((0,), dtype=np.float64)
                else:
                    f_transform_func = field_transform_func_dict[field_name]
                    comp_dict[field_nice_name] = f_transform_func(field_val_arr)
            return comp_dict

        ell1_dict_list = \
            [form_comp_dict(ell_dict, field_nice_names1_dict, field_transform_func1_dict)
             for ell_dict in list(dict_ell1_list)]
        ell2_dict_list = \
            [form_comp_dict(ell_dict, field_nice_names2_dict, field_transform_func2_dict)
             for ell_dict in list(dict_ell2_list)]
        ell1_dict_arr = np.reshape(np.array(ell1_dict_list), first_shape_vec)
        ell2_dict_arr = np.reshape(np.array(ell2_dict_list), sec_shape_vec)
        if not (isn_first_scalar and isn_sec_scalar):
            if isn_first_scalar:
                ell2_dict_arr = np.tile(ell2_dict_arr, list(first_shape_vec))
            else:
                ell1_dict_arr = np.tile(ell1_dict_arr, list(sec_shape_vec))
        is_eq_arr, report_str = dict_compare_vec(ell1_dict_arr, ell2_dict_arr, abs_tol, rel_tol)
        if isn_first_scalar:
            is_eq_arr = np.reshape(is_eq_arr, first_shape_vec)
        else:
            is_eq_arr = np.reshape(is_eq_arr, sec_shape_vec)
        return is_eq_arr, report_str

    # noinspection PyUnusedLocal
    @classmethod
    def eq(cls, ell_first_arr: np.ndarray, ell_second_arr: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, str]:
        return cls.is_equal(ell_first_arr, ell_second_arr)

    def __eq__(self, other) -> bool:
        is_eq, _ = self._is_equal_internal(self, other, False)
        return np.array(is_eq).flatten()[0]

    @classmethod
    def is_equal(cls, ell_first_arr, ell_sec_arr, is_prop_included: bool = False) -> Tuple[np.ndarray, str]:
        return cls._is_equal_internal(ell_first_arr, ell_sec_arr, is_prop_included)

    @classmethod
    def get_copy(cls, ell_arr: np.ndarray) -> np.ndarray:
        cls._check_is_me_virtual(ell_arr)
        ell_arr = np.array(ell_arr)
        n_elems = ell_arr.size
        if n_elems == 0:
            return np.copy(ell_arr)
        ell_shape_vec = ell_arr.shape
        ell_arr = ell_arr.flatten()
        if ell_arr.size == 1:
            # noinspection PyProtectedMember
            return np.array(ell_arr[0]._get_single_copy())
        else:
            copy_ell_arr = np.empty((n_elems,), dtype=np.object)
            for i_elem in range(n_elems):
                # noinspection PyProtectedMember
                copy_ell_arr[i_elem] = ell_arr[i_elem]._get_single_copy()
            return np.reshape(copy_ell_arr, ell_shape_vec)

    @classmethod
    def is_empty(cls, ell_arr: np.ndarray):
        cls._check_is_me_virtual(ell_arr)
        return cls.dimension(ell_arr) == 0

    def rep_mat(self, shape_vec: Union[list, tuple, np.ndarray]) -> np.ndarray:
        shape_vec = np.array(shape_vec)
        if not is_numeric(shape_vec):
            throw_error('wrongInput:shape_vec', 'size array should be numeric')
        shape_vec = try_treat_as_real(shape_vec)
        if shape_vec.size == 0 or shape_vec.size != np.max(shape_vec.shape):
            throw_error('wrongInput:shape_vec', 'size vector must have at least two elements and be not a matrix')
        shape_vec = shape_vec.flatten()
        if not np.all(np.logical_and(shape_vec == np.fix(shape_vec), shape_vec >= 0)):
            throw_error('wrongInput:shape_vec', 'size vector must contain non-negative integer values')
        n_elems = np.prod(shape_vec).flatten()[0]
        shape_vec = tuple(shape_vec)
        ell_arr = np.empty((n_elems,), dtype=np.object)
        for i_elem in range(n_elems):
            ell_arr[i_elem] = self._get_single_copy()
        ell_arr = np.reshape(ell_arr, shape_vec)
        return ell_arr
