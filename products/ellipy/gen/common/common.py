import inspect
import warnings
from typing import Tuple, Union, Callable
import numpy as np


def get_caller_name_ext(skip=1) -> Tuple[str, str]:
    """Get a name of a caller as name of function/method and module.class
    
       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
       
       Empty strings are returned if skipped levels exceed stack height
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
        return '', ''
    parentframe = stack[start][0]    
    
    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename == '<module>':  # top level usually
        codename = ''
    del parentframe
    return codename, ".".join(name)


def get_caller_name(skip=1, mode: str ='default') -> str:
    """Get a name of a caller as name of module.class.method
    
       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
       
       An empty string is returned if skipped levels exceed stack height
    """
    code_name, class_name = get_caller_name_ext(skip + 1)
    if mode == 'default':
        if class_name == '':
            return code_name
        else:
            return class_name
    elif mode == 'full':
        if class_name == '':
            return code_name
        else:
            return class_name + '.' + code_name
    else:
        code_name, class_name = get_caller_name_ext()
        raise ValueError((class_name + '.' + code_name).upper() + 'wrongInput: Unknown mode: {}'.format(mode))    


def throw_error(tag: str, message: str, n_caller_stack_steps_up: int = 1, throw: bool = True) -> \
        Union[BaseException, None]:
    e = ValueError(get_caller_name(n_caller_stack_steps_up + 1, 'full').replace(
        '.', ':').upper() + ':' + tag + ':' + message)
    if throw:
        raise e
    else:
        return e


def throw_warn(tag: str, message: str) -> None:
    warnings.warn(get_caller_name(2, 'full').replace('.', ':').upper() + ':' + tag + ':' + message,
                  RuntimeWarning, stacklevel=2)


def is_member(a_vec, b_vec) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(a_vec, np.ndarray):
        a_vec = np.array(a_vec)
    bool_ind_vec = np.isin(a_vec, b_vec)
    common_vec = a_vec[bool_ind_vec]
    common_unique_vec, common_inv_vec = np.unique(common_vec, return_inverse=True)
    b_unique_vec, b_ind_vec = np.unique(b_vec, return_index=True)
    common_ind_vec = b_ind_vec[np.isin(b_unique_vec, common_unique_vec, assume_unique=True)]
    return bool_ind_vec, common_ind_vec[common_inv_vec]


def is_numeric(array: np.ndarray) -> bool:
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Parameters
    ----------
    array : array-like
        The array to check.

    Returns
    -------
    is_numeric : `bool`
        True if the array has a numeric datatype, False if not.

    """
    _NUMERIC_KINDS = set('buifc')
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS


def abs_rel_compare(left_arr: np.ndarray, right_arr: np.ndarray,
                    abs_tol: float, rel_tol: Union[float, None],
                    f_norm_op: Callable[[np.ndarray], np.ndarray]) -> \
        Tuple[bool, float, bool, float, float, str]:
    __FORMAT_SPEC = '%.17g'
 
    if not (left_arr.shape == right_arr.shape and is_numeric(left_arr) and is_numeric(right_arr)):
        throw_error('wrongInput:wrongArgs', 'left_arr and right_arr must be ' +
                    'numeric arrays with the same size')
    if not (abs_tol is not None and np.isscalar(abs_tol) and np.isreal(abs_tol) and abs_tol >= 0.):
        throw_error('wrongInput:wrongAbsTol', 'abs_tol must be a nonnegative scalar')
    if rel_tol is not None and not (np.isscalar(rel_tol) and np.isreal(rel_tol) and rel_tol >= 0.):
        throw_error('wrongInput:wrongRelTol', 'rel_tol must be either None or a nonnegative scalar')
    if not callable(f_norm_op):
        throw_error('wrongInput:wrongNormOp', 'f_norm_op must be a function')

    diff_arr = f_norm_op(left_arr - right_arr)
    if diff_arr.size == 0:
        abs_diff = np.array([])
        abs_r_diff = abs_diff
        is_equal = True
    else:
        abs_diff = np.amax(diff_arr)
        abs_r_diff = abs_diff
        is_equal = abs_r_diff <= abs_tol

    rel_diff = np.array([])
    abs_m_rel_diff = rel_diff

    if rel_tol is not None and diff_arr.size > 0:
        is_rel_diff_triggered_arr = diff_arr > abs_tol
        is_rel_diff_triggered = np.any(is_rel_diff_triggered_arr)
        if is_rel_diff_triggered:
            arg_sum_norm_arr = f_norm_op(left_arr) + f_norm_op(right_arr)
            is_rel_diff_triggered_arr = np.logical_and(is_rel_diff_triggered_arr, arg_sum_norm_arr > abs_tol)
            is_rel_diff_triggered = np.any(is_rel_diff_triggered_arr)
            if is_rel_diff_triggered:
                temp_arr = np.zeros(diff_arr.shape, dtype=np.float64)
                temp_2_arr = np.array(temp_arr, copy=True)
                temp_arr[is_rel_diff_triggered_arr] = \
                    2. * diff_arr[is_rel_diff_triggered_arr] / \
                    arg_sum_norm_arr[is_rel_diff_triggered_arr]
                rel_diff = np.amax(temp_arr)
                temp_2_arr[temp_arr == rel_diff] = diff_arr[temp_arr == rel_diff]
                abs_m_rel_diff = np.amax(temp_2_arr)
                temp_arr = np.zeros(diff_arr.shape, dtype=np.float64)
                isn_rel_diff_triggered_arr = np.logical_not(is_rel_diff_triggered_arr)
                temp_arr[isn_rel_diff_triggered_arr] = diff_arr[isn_rel_diff_triggered_arr]
                abs_r_diff = np.amax(temp_arr)
        if rel_diff.size > 0:
            is_equal = abs_r_diff <= abs_tol and rel_diff <= rel_tol
    else:
        is_rel_diff_triggered = False

    if is_equal:
        report_str = ''
    else:
        if is_rel_diff_triggered:
            report_str = (('relative difference (FORMAT_SPEC) is greater' +
                           ' than the specified tolerance (FORMAT_SPEC); absolute' +
                           ' difference (FORMAT_SPEC), absolute tolerance (FORMAT_SPEC)').replace(
                'FORMAT_SPEC', __FORMAT_SPEC) % (rel_diff, rel_tol, abs_m_rel_diff, abs_tol))
        else:
            report_str = (('absolute difference (FORMAT_SPEC) is greater' +
                           ' than the specified tolerance (FORMAT_SPEC)').replace(
                'FORMAT_SPEC', __FORMAT_SPEC) % (abs_diff, abs_tol))
    return is_equal, abs_diff, is_rel_diff_triggered, rel_diff, abs_m_rel_diff, report_str
