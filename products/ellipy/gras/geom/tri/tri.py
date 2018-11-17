from typing import Tuple, Dict, Callable
import numpy as np


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
        [1, 2, 3],  [1, 3, 4],  [1, 4, 5],   [1, 5, 6],    [1, 6, 2],
        [2, 7, 3],  [3, 8, 4],  [4, 9, 5],   [5, 10, 6],   [6, 11, 2],
        [7, 8, 3],  [8, 9, 4],  [9, 10, 5],  [10, 11, 6],  [11, 7, 2],
        [7, 12, 8], [8, 12, 9], [9, 12, 10], [10, 12, 11], [11, 12, 7]
    ]))
    return v_mat, f_mat


def map_face_2_edge(v_mat: np.ndarray, f_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass

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
