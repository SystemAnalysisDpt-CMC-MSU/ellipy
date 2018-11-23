import numpy as np
import re


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


def var_replace(m_mat: np.ndarray, from_var_name: str, to_var_name: str) -> np.ndarray:
    pass
