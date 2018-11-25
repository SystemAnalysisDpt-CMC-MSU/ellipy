import numpy as np
import re
from ellipy.gen.common.common import throw_error


def is_dependent(m_mat: np.ndarray, is_discrete: bool = False) -> bool:
    if is_discrete:
        reg_expression = b'(^k\b|k|^k$|\bk$)'
    else:
        reg_expression = b'(^t\b|t|^t$|\bt$)'
    arr = np.concatenate(m_mat).astype('str')  # realisation of cell2mat
    reg_arr = re.search(reg_expression, arr, flags=re.MULTILINE)  # = regexp
    if reg_arr:
        is_depend = True
    else:
        is_depend = False
    return is_depend


def norm(s):
    s = s.replace('+', '\+')
    s = s.replace('*', '\*')
    return s


def var_replace(m_mat: np.ndarray, from_var_name: str, to_var_name: str) -> np.ndarray:

    if not m_mat:
        throw_error('wrongInput: m_mat', 'mCMat must not be empty')
    if not isinstance(from_var_name, str):
        throw_error('wrongInput: from_var_name', 'from_var_name is expected to be a string')
    if not isinstance(to_var_name, str):
        throw_error('wrongInput: to_var_name', 'to_var_name is expected to be a string')

    for i in range(len(m_mat)):
        for j in range(len(m_mat[i])):
            if ~isinstance(m_mat[i][j], str):
                m_mat[i][j] = str(m_mat[i][j])
    from_var = norm(from_var_name)
    reg_expr = '\W' + from_var + '\W|^' + from_var + '\W|\W' + from_var + '$'
    for i in range(len(m_mat)):
        for j in range(len(m_mat[i])):
            m_mat[i][j] = re.sub(reg_expr, to_var_name, m_mat[i][j], flags=re.MULTILINE)
    return(m_mat)
