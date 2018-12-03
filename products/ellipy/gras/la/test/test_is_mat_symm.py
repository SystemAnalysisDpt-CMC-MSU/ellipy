import numpy as np
from ellipy.gras.la.la import is_mat_symm
from numpy import diag
from numpy.random import rand
from math import inf as INF
import pytest
from ellipy.gen.common.common import throw_error, is_numeric


def test_is_mat_symm():
    # assert is_mat_symm(2) //wrong input!

    assert is_mat_symm(np.diagflat(np.arange(5) + 1))

    test_mat = rand(20, 20)
    assert is_mat_symm(test_mat @ (test_mat.T))

    test_mat = 10 * rand(100, 100)
    assert is_mat_symm(test_mat + (test_mat.T))

    __ABS_TOL = 1e-7
    test_mat = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0.6912, 0.2368, 1.7280],
                         [0, 0, 0, 0.2368, 0.1552, 0.5920],
                         [0, 0, 0, 1.7280, 0.5920, 4.3200]])
    test_mat[4, 5] = test_mat[4, 5] + __ABS_TOL / 1000
    assert is_mat_symm(test_mat, __ABS_TOL)

    test_mat = np.array([[2, 1], [3, 2]])
    assert not is_mat_symm(test_mat)

    test_mat = 10 * rand(20, 20) + np.diagflat(np.arange(19) + 1, 1)
    assert not is_mat_symm(test_mat)

    # Wrong input!
    # with pytest.raises(Exception) as e:
    #     is_mat_symm(np.eye(5, 7))
    # assert 'wrongInput:non_square_mat' in str(e.value)

    # Assert will return False!
    # test_mat = np.diagflat([INF, -INF, 1, 2, 0])
    # assert is_mat_symm(test_mat)
