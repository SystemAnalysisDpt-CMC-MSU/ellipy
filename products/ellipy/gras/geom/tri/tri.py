from typing import Tuple, Dict, Callable
import numpy as np
from ellipy.gen.common.common import throw_error


def ell_tube_2_tri(n_e_points: int, n_points: int) -> np.ndarray:
    pass


def ell_tube_discr_tri(n_dim: int, m_dim: int) -> np.ndarray:
    pass


def icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    __IND_VEC = np.array(np.arange(0, 5))
    __Z_VEC = np.array([0.5] * 5)
    pi = np.pi
    tau = (np.sqrt(5.0) + 1) / 2
    r = tau - 0.5
    v_mat = np.ndarray(shape=(12, 3), dtype=float, buffer=np.zeros(shape=(12, 3)))
    v_mat[0][2] = 1.0
    v_mat[11][2] = -1.0
    #
    alpha_vec = -pi / 5 + __IND_VEC * pi/2.5
    v_mat[1+__IND_VEC] = np.column_stack((np.cos(alpha_vec)/r, np.sin(alpha_vec)/r, __Z_VEC/r))
    #
    alpha_vec = __IND_VEC * pi/2.5
    v_mat[6+__IND_VEC] = np.column_stack((np.cos(alpha_vec)/r, np.sin(alpha_vec)/r, -__Z_VEC/r))
    f_mat = np.ndarray(shape=(20, 3), dtype=int, buffer=np.array([
        [0, 1, 2],  [0, 2, 3],  [0, 3, 4],  [0, 4, 5],   [0, 5, 1],
        [1, 6, 2],  [2, 7, 3],  [3, 8, 4],  [4, 9, 5],   [5, 10, 1],
        [6, 7, 2],  [7, 8, 3],  [8, 9, 4],  [9, 10, 5],  [10, 6, 1],
        [6, 11, 7], [7, 11, 8], [8, 11, 9], [9, 11, 10], [10, 11, 6]
    ]))
    return v_mat, f_mat


def map_face_2_edge(f_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.size(f_mat, 1) != 3:
        throw_error('wrongInput:f_mat', 'The number of columns should be equal to 3')
    n_faces = np.size(f_mat, 0)
    ind_2_check = np.array([[0, 1], [0, 2], [1, 2]])
    pos_edges = np.sort(np.reshape(f_mat[:, ind_2_check], newshape=(-1, 2)))
    ind_f_vec = np.repeat(np.arange(n_faces), 3)
    e_mat, inv_idx = np.unique(pos_edges, axis=0, return_inverse=True)
    sort_idx_vec = np.argsort(inv_idx)
    inv_idx = inv_idx[sort_idx_vec]
    ind_f_vec = ind_f_vec[sort_idx_vec]
    ind_shift_vec = np.bincount(inv_idx)
    n_edges = np.size(e_mat, 0)
    ind_f2e_vec = np.zeros(shape=(np.sum(ind_shift_vec), 1))
    ind_f2e_vec[0] = 1
    ind_f2e_vec[np.cumsum(ind_shift_vec[:-1])] = np.ones(shape=(n_edges-1, 1))
    ind_f2e_vec = np.cumsum(ind_f2e_vec) - 1
    f_mat = f_mat[ind_f_vec]
    ind_edge_num_vec = \
        + 1 * np.all(np.equal(f_mat[:, [0, 1]], e_mat[ind_f2e_vec]), 1) \
        - 1 * np.all(np.equal(f_mat[:, [1, 0]], e_mat[ind_f2e_vec]), 1) \
        + 2 * np.all(np.equal(f_mat[:, [1, 2]], e_mat[ind_f2e_vec]), 1) \
        - 2 * np.all(np.equal(f_mat[:, [2, 1]], e_mat[ind_f2e_vec]), 1) \
        + 3 * np.all(np.equal(f_mat[:, [0, 2]], e_mat[ind_f2e_vec]), 1) \
        - 3 * np.all(np.equal(f_mat[:, [2, 0]], e_mat[ind_f2e_vec]), 1)
    ind_sort_vec = np.lexsort((ind_f_vec, np.abs(ind_edge_num_vec)-1))
    ind_f2e_vec = ind_f2e_vec[ind_sort_vec]
    f2e_mat = np.reshape(ind_f2e_vec, newshape=(3, n_faces)).T
    f2e_is_dir_mat = np.reshape(np.greater(ind_edge_num_vec[ind_sort_vec], 0), newshape=(3, n_faces)).T
    return e_mat, f2e_mat, f2e_is_dir_mat


def shrink_face_tri(v_mat: np.ndarray, f_mat: np.ndarray,
                    max_edge_len: float, n_max_steps: float = np.inf,
                    f_vert_adjust_func: Callable[[np.ndarray], np.ndarray] = None) -> \
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    def deal(inp_arr: np.ndarray) -> np.ndarray:
        return inp_arr

    if f_vert_adjust_func is None:
        f_vert_adjust_func = deal


def sphere_tri(depth: int) -> Tuple[np.ndarray, np.ndarray]:
    pass


def sphere_tri_ext(n_dim: int, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    pass
