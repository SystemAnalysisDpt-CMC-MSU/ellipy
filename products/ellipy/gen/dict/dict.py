from typing import Tuple, List, Union
from ellipy.gen.common.common import throw_error, abs_rel_compare, is_numeric
import numpy as np


def dict_compare(dict_x: Union[np.ndarray, List[dict], dict],
                 dict_y: Union[np.ndarray, List[dict], dict],
                 abs_tol: float = 0., rel_tol: float = None) -> Tuple[bool, str]:
    x_shape_vec = None
    y_shape_vec = None
    if type(dict_x) in [np.ndarray, list, dict]:
        if type(dict_x) == dict:
            dict_x = np.array([dict_x])
        else:
            dict_x = np.array(dict_x)
            if dict_x.size > 0:
                if type(dict_x.flatten()[0]) != dict:
                    throw_error('wrongInput:dict_x',
                                'dict_x is expected to be dictionary or array or list of dictionaries')
        x_shape_vec = dict_x.shape
    else:
        throw_error('wrongInput:dict_x',
                    'dict_x is expected to be dictionary or array or list of dictionaries')
    if type(dict_y) in [np.ndarray, list, dict]:
        if type(dict_y) == dict:
            dict_y = np.array([dict_y])
        else:
            dict_y = np.array(dict_y)
            if dict_y.size > 0:
                if type(dict_y.flatten()[0]) != dict:
                    throw_error('wrongInput:dict_y',
                                'dict_y is expected to be dictionary or array or list of dictionaries')
        y_shape_vec = dict_y.shape
    else:
        throw_error('wrongInput:dict_y',
                    'dict_y is expected to be dictionary or array or list of dictionaries')
    if x_shape_vec != y_shape_vec:
        return False, 'sizes are different'
    is_eq_vec, report_str = dict_compare_vec(dict_x.flatten(), dict_y.flatten(), abs_tol, rel_tol)
    return np.all(is_eq_vec), report_str


def dict_compare_vec(dict_x_arr: np.ndarray, dict_y_arr: np.ndarray,
                     abs_tol: float = 0., rel_tol: float = None) -> Tuple[np.ndarray, str]:
    if dict_x_arr.shape != dict_y_arr.shape:
        is_eq = np.array(False)
        report_str = 'sizes are different'
        return is_eq, report_str

    def dict_compare_1d_list(dict_x_list: List[dict], dict_y_list: List[dict],
                             inp_abs_tol: float, inp_rel_tol: float) -> Tuple[List[bool], List[str]]:
        def dict_compare_scalar(inp_dict_x: dict, inp_dict_y: dict,
                                sc_abs_tol: float, sc_rel_tol: float) -> Tuple[bool, List[str]]:
            if not (type(inp_dict_x) == dict and type(inp_dict_y) == dict):
                throw_error('wrongInput', 'both inputs are expected to be dictionaries')

            sc_report_str_list = []
            key_x_set = set(inp_dict_x.keys())
            key_y_set = set(inp_dict_y.keys())
            is_sc_eq = key_x_set == key_y_set
            if not is_sc_eq:
                key_x_minus_y_list = sorted(list(key_x_set - key_y_set))
                key_y_minus_x_list = sorted(list(key_y_set - key_x_set))
                sc_report_str_list.append('Field names are different, left-right:{}, right-left: {}'.format(
                    '|'.join(key_x_minus_y_list), '|'.join(key_y_minus_x_list)))
                return False, sc_report_str_list

            def comp_fun(x, y, comp_abs_tol: float, comp_rel_tol: float) -> Tuple[bool, str]:
                comp_report_str = ''

                x_dtype = np.asarray(x).dtype.kind
                y_dtype = np.asarray(y).dtype.kind
                x_type = type(x)

                if x_type != type(y) or x_dtype != y_dtype:
                    comp_report_str = 'Different types'
                    return False, comp_report_str

                if is_numeric(x) and x_type != list:
                    x_arr = np.array(x)
                    y_arr = np.array(y)
                    x_shape_vec = x_arr.shape
                    y_shape_vec = y_arr.shape
                    if x_shape_vec != y_shape_vec:
                        comp_report_str = 'Different sizes (left: {}, right: {})'.format(x_shape_vec, y_shape_vec)
                        return False, comp_report_str
                    if x_dtype in 'ui':
                        x_arr = np.array(x_arr, dtype=np.float)
                        y_arr = np.array(y_arr, dtype=np.float)
                    if x_dtype == 'fc':
                        if not np.array_equal(np.isnan(x_arr), np.isnan(y_arr)):
                            comp_report_str = 'Nans are on the different places'
                            return False, comp_report_str
                        if not np.array_equal(x_arr == -np.inf, y_arr == -np.inf):
                            comp_report_str = '-Infs are on the different places'
                            return False, comp_report_str
                        if not np.array_equal(x_arr == np.inf, y_arr == np.inf):
                            comp_report_str = '+Infs are on the different places'
                            return False, comp_report_str
                        is_comp_arr = np.isfinite(x_arr)
                    else:
                        is_comp_arr = np.ones(x_arr.shape, dtype=bool)
                    is_comp_eq, _, _, _, _, comp_report_str = abs_rel_compare(
                        x_arr[is_comp_arr], y_arr[is_comp_arr], comp_abs_tol, comp_rel_tol, lambda z: np.abs(z))
                    if not is_comp_eq:
                        comp_report_str = 'Max. ' + comp_report_str
                        return False, comp_report_str
                elif x_type in [dict, list, np.ndarray]:
                    is_dict = x_type == dict
                    if not is_dict and x_type == np.ndarray and x.size > 0:
                        is_dict = type(x.flatten()[0]) == dict
                    if is_dict:
                        is_comp_eq, comp_report_str = dict_compare(x, y, comp_abs_tol, comp_rel_tol)
                        if not is_comp_eq:
                            return False, comp_report_str
                    else:
                        n_comp_elems = len(x)
                        if n_comp_elems == 0:
                            return True, ''
                        is_comp_eq = True
                        for i_comp_elem in range(n_comp_elems):
                            is_comp_eq, comp_report_str = comp_fun(x[i_comp_elem], y[i_comp_elem],
                                                                   comp_abs_tol, comp_rel_tol)
                            if not is_comp_eq:
                                comp_report_str = '{' + str(i_comp_elem) + '}' + comp_report_str
                                break
                        if not is_comp_eq:
                            return False, comp_report_str
                elif x != y:
                    return False, 'values are different'
                return True, comp_report_str

            key_list = sorted(list(key_x_set))
            for key_name in key_list:
                inp_x = inp_dict_x[key_name]
                is_cur_eq, cur_report_str = comp_fun(inp_x, inp_dict_y[key_name],
                                                     sc_abs_tol, sc_rel_tol)
                is_sc_eq &= is_cur_eq
                if not is_cur_eq:
                    if type(inp_x) == dict:
                        cur_report_str = '.' + key_name + cur_report_str
                    else:
                        cur_report_str = '.' + key_name + '--> ' + cur_report_str
                    sc_report_str_list.append(cur_report_str)
            return is_sc_eq, sc_report_str_list

        is_out_eq_list, out_report_str_list_list = \
            zip(*[dict_compare_scalar(dict_x, dict_y, inp_abs_tol, inp_rel_tol) for
                  (dict_x, dict_y) in zip(dict_x_list, dict_y_list)])
        is_out_eq_list = list(is_out_eq_list)
        out_report_str_list_list = list(out_report_str_list_list)
        out_report_str_list = []
        for i_out_elem in range(len(is_out_eq_list)):
            if not is_out_eq_list[i_out_elem]:
                out_report_str_list += ['({}){}'.format(i_out_elem, out_report_str)
                                        for out_report_str in out_report_str_list_list[i_out_elem]]

        return is_out_eq_list, out_report_str_list

    is_eq_list, report_str_list = dict_compare_1d_list(
        list(dict_x_arr.flatten()), list(dict_y_arr.flatten()), abs_tol, rel_tol)
    n_reports = len(report_str_list)
    if n_reports > 0:
        report_str = '\n'.join(report_str_list)
    else:
        report_str = ''
    return np.array(is_eq_list), report_str
