import numpy as np
import re
from ellipy.gen.common.common import throw_error


def is_dependent(m_mat: np.ndarray, is_discrete: bool = False) -> bool:
    m_mat = np.vectorize(str)(m_mat)
    if is_discrete:
        reg_arr = np.vectorize(re.search)(r'\W' + 'k' + r'\W', m_mat)
    else:
        reg_arr = np.vectorize(re.search)(r'\W' + 't' + r'\W', m_mat)
    if reg_arr.any():
        is_depend = True
    else:
        is_depend = False
    return is_depend


def var_replace(m_mat: np.ndarray, from_var_name: str, to_var_name: str) -> np.ndarray:
    if not m_mat.size:
        throw_error('wrongInput: m_mat', 'm_mat must not be empty')
    if not isinstance(from_var_name, str):
        throw_error('wrongInput: from_var_name', 'from_var_name is expected to be a string')
    if not isinstance(to_var_name, str):
        throw_error('wrongInput: to_var_name', 'to_var_name is expected to be a string')
    m_mat = np.vectorize(str)(m_mat)  # Векторизуем функцию str() и применяем ко всему массиву. В пару раз быстрее будет
    to_var_name = '(' + to_var_name + ')'

    def _replace(elem, to_replace, value):
        return elem.replace(to_replace, value)
    m_mat = np.vectorize(_replace)(m_mat, ' ', '')
    m_mat = np.vectorize(re.sub)(r'\b' + from_var_name + r'\b', to_var_name, m_mat)
    return m_mat
