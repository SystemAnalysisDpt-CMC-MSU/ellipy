from typing import Tuple, Dict, Callable, Union, List
import numpy as np
from numpy import matlib as ml
from ellipy.gen.common.common import throw_error, is_member, is_numeric
from ellipy.gras.gen.gen import sort_rows_tol
from ellipy.gras.geom.geom import circle_part


def ell_tube_2_tri(n_e_points: int, n_points: int) -> np.ndarray:
    pass


def ell_tube_discr_tri(n_dim: int, m_dim: int) -> np.ndarray:
    pass


def icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    __IND_VEC = np.arange(0, 5)
    __Z_VEC = np.full((5,), 0.5)
    tau = (np.sqrt(5.0) + 1) / 2
    r = tau - 0.5
    v_mat = np.zeros(shape=(12, 3), dtype=np.float64)
    v_mat[0][2] = 1.0
    v_mat[11][2] = -1.0
    #
    alpha_vec = -np.pi / 5 + __IND_VEC * np.pi / 2.5
    v_mat[1+__IND_VEC] = np.column_stack((np.cos(alpha_vec)/r, np.sin(alpha_vec)/r, __Z_VEC/r))
    #
    alpha_vec = __IND_VEC * np.pi / 2.5
    v_mat[6+__IND_VEC] = np.column_stack((np.cos(alpha_vec)/r, np.sin(alpha_vec)/r, -__Z_VEC/r))
    f_mat = np.array([
        [0, 1, 2],  [0, 2, 3],  [0, 3, 4],  [0, 4, 5],   [0, 5, 1],
        [1, 6, 2],  [2, 7, 3],  [3, 8, 4],  [4, 9, 5],   [5, 10, 1],
        [6, 7, 2],  [7, 8, 3],  [8, 9, 4],  [9, 10, 5],  [10, 6, 1],
        [6, 11, 7], [7, 11, 8], [8, 11, 9], [9, 11, 10], [10, 11, 6]
    ], dtype=int)
    return v_mat, f_mat


def get_face_edge_info(f_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if f_mat.ndim == 1:
        if np.shape(f_mat) != (3,):
            throw_error('wrongInput:f_mat', 'The number of columns should be equal to 3')
        else:
            f_mat = np.reshape(f_mat, newshape=(1, 3))
    elif np.size(f_mat, 1) != 3:
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
    return f_mat, e_mat, ind_f_vec, ind_shift_vec


def map_face_2_edge(f_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_mat, e_mat, ind_f_vec, ind_shift_vec = get_face_edge_info(f_mat)
    n_faces = np.size(f_mat, 0)
    n_edges = np.size(e_mat, 0)
    ind_f2e_vec = np.zeros(shape=(np.sum(ind_shift_vec), 1), dtype=np.int32)
    ind_f2e_vec[0] = 1
    ind_f2e_vec[np.cumsum(ind_shift_vec[:-1])] = np.ones(shape=(n_edges-1, 1))
    ind_f2e_vec = np.cumsum(ind_f2e_vec) - 1
    f_mat = f_mat[ind_f_vec]
    ind_edge_num_vec = \
        (+ 1 * np.all(np.equal(f_mat[:, [0, 1]], e_mat[ind_f2e_vec]), 1)
         - 1 * np.all(np.equal(f_mat[:, [1, 0]], e_mat[ind_f2e_vec]), 1)
         + 2 * np.all(np.equal(f_mat[:, [1, 2]], e_mat[ind_f2e_vec]), 1)
         - 2 * np.all(np.equal(f_mat[:, [2, 1]], e_mat[ind_f2e_vec]), 1)
         + 3 * np.all(np.equal(f_mat[:, [0, 2]], e_mat[ind_f2e_vec]), 1)
         - 3 * np.all(np.equal(f_mat[:, [2, 0]], e_mat[ind_f2e_vec]), 1)).flatten()
    ind_sort_vec = np.lexsort((ind_f_vec, np.abs(ind_edge_num_vec)-1))
    ind_f2e_vec = ind_f2e_vec[ind_sort_vec]
    if n_faces > 1:
        f2e_mat = np.reshape(ind_f2e_vec, newshape=(3, n_faces)).T
        f2e_is_dir_mat = np.reshape(np.greater(ind_edge_num_vec[ind_sort_vec], 0), newshape=(3, n_faces)).T
    else:
        f2e_mat = ind_f2e_vec
        f2e_is_dir_mat = np.greater(ind_edge_num_vec[ind_sort_vec], 0)
    return e_mat, f2e_mat, f2e_is_dir_mat


def shrink_face_tri(v_mat: np.ndarray, f_mat: np.ndarray,
                    max_edge_len: float, n_max_steps: float = np.inf,
                    f_vert_adjust_func: Callable[[np.ndarray], np.ndarray] = None,
                    is_stat_collected: bool = False) -> \
        Union[Tuple[np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray,
                    Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
    def deal(inp_arr: np.ndarray) -> np.ndarray:
        return inp_arr

    if f_vert_adjust_func is None:
        f_vert_adjust_func = deal
        is_adjust_func_spec = False
    else:
        is_adjust_func_spec = True

    # Build Face to Edges map and edge orientation map for each face
    e_mat, f2e_mat, f2e_is_dir_mat = map_face_2_edge(f_mat)

    n_verts = np.size(v_mat, 0)
    n_edges = np.size(e_mat, 0)
    if f_mat.ndim == 1:
        n_faces = 1
    else:
        n_faces = np.size(f_mat, 0)

    # Build edge distances
    d_mat = v_mat[e_mat[:, 0]] - v_mat[e_mat[:, 1]]
    e_length_vec = np.sqrt(np.sum(d_mat * d_mat, 1))

    # Set up statistics collection
    n_vert_vec = n_verts
    n_edge_vec = n_edges
    n_face_vec = n_faces
    n_edges_to_shrink_vec = np.zeros(shape=(0, 1))
    max_edge_length_vec = np.zeros(shape=(0, 1))

    i_step = 1
    while True:
        # Find edges that need to be shortened
        is_e2orig_part_vec = e_length_vec > max_edge_len
        if np.any(is_e2orig_part_vec) and (i_step <= n_max_steps):
            # Find faces that need to be shortened
            if n_faces > 1:
                is_f2part_vec = np.any(is_e2orig_part_vec[f2e_mat], 1)
            else:
                # this is because is_e2part_vec(f2e_mat) produces a column-vector for one face
                is_f2part_vec = np.any(is_e2orig_part_vec[f2e_mat])

            # Read just indices of partitioned edges
            f2e_part_mat = f2e_mat[is_f2part_vec]
            f2e_is_dir_part_mat = f2e_is_dir_mat[is_f2part_vec]
            is_e2part_vec = is_e2orig_part_vec
            is_e2part_vec[f2e_part_mat] = True

            # Build indices of partitioned vertices
            ind_v1_vec = e_mat[is_e2part_vec, 0]
            ind_v2_vec = e_mat[is_e2part_vec, 1]

            v_new_mat = (v_mat[ind_v1_vec] + v_mat[ind_v2_vec]) * 0.5
            n_new_verts = np.size(v_new_mat, 0)

            # Collect stats
            if is_stat_collected:
                n_edges_to_shrink_vec = np.array(np.append(n_edges_to_shrink_vec, np.sum(is_e2part_vec)),
                                                 dtype=np.int32)
                max_edge_length_vec = np.append(max_edge_length_vec, np.max(e_length_vec))

            # Find new faces, edges and vertices
            n_shrinked_faces = np.sum(is_f2part_vec).flatten()[0]

            # Build indices of new vertices for each partitioned face
            ind_e_part_vec = np.cumsum(is_e2part_vec) + n_verts - 1
            ind_vf12_vec = ind_e_part_vec[f2e_part_mat[:, 0]]
            ind_vf23_vec = ind_e_part_vec[f2e_part_mat[:, 1]]
            ind_vf13_vec = ind_e_part_vec[f2e_part_mat[:, 2]]

            # Build indices of vertices of partitioned faces
            ind_vf1_vec = f_mat[is_f2part_vec, 0]
            ind_vf2_vec = f_mat[is_f2part_vec, 1]
            ind_vf3_vec = f_mat[is_f2part_vec, 2]

            # Remove partitioned faces and edges
            f_mat = f_mat[~is_f2part_vec]
            f2e_mat = f2e_mat[~is_f2part_vec]
            f2e_is_dir_mat = f2e_is_dir_mat[~is_f2part_vec]
            is_e_kept_vec = np.zeros((n_edges,), dtype=bool)
            is_e_kept_vec[f2e_mat] = True
            ind_e_kept_vec = np.cumsum(is_e_kept_vec) - 1
            f2e_mat = ind_e_kept_vec[f2e_mat]
            e_mat = e_mat[is_e_kept_vec]
            e_length_vec = e_length_vec[is_e_kept_vec]
            n_edges = np.size(e_mat, 0)

            # Build the first group of new edges (internal edges)
            e1_new_mat = np.vstack([
                np.column_stack([ind_vf12_vec, ind_vf23_vec]),
                np.column_stack([ind_vf12_vec, ind_vf13_vec]),
                np.column_stack([ind_vf23_vec, ind_vf13_vec])])

            # Build the first group of new faces (internal faces)
            f1_new_mat = np.column_stack([ind_vf12_vec, ind_vf23_vec, ind_vf13_vec])

            ind_new_e_vec = np.arange(n_edges, n_edges + n_shrinked_faces)

            # Build face-to-edge map for the first group of new edges and faces
            f2e1_new_mat = (ml.repmat(ind_new_e_vec, 3, 1).T +
                            np.ones(shape=(n_shrinked_faces, 1)) @ (np.matrix([0, 2, 1]) * n_shrinked_faces))

            # Build face-to-edge directions for the first group of new edges and faces
            f2e1_is_dir_new_mat = np.ones(shape=np.shape(f2e1_new_mat), dtype=bool)

            # Build the second group of new edges (edges on the boundaries of the partitioned faces)
            ind_new_vert = np.arange(n_verts, n_verts + n_new_verts)
            e2_new_mat = np.vstack([
                np.column_stack([ind_v1_vec, ind_new_vert]),
                np.column_stack([ind_new_vert, ind_v2_vec])
            ])

            # Build the second group of new faces (faces on the boundaries of partitioned faces)
            f2_new_mat = np.vstack([
                np.column_stack([ind_vf12_vec, ind_vf13_vec, ind_vf1_vec]),
                np.column_stack([ind_vf23_vec, ind_vf12_vec, ind_vf2_vec]),
                np.column_stack([ind_vf13_vec, ind_vf23_vec, ind_vf3_vec])
            ])

            # Build face-to-edge map for the second group of faces and edges
            ind_shift_edge_vec = np.arange(n_shrinked_faces).T

            dir_mat = ~np.logical_xor(np.kron(np.array([[0, 0], [1, 0], [1, 1]]) * n_shrinked_faces,
                                              np.ones(shape=(n_shrinked_faces, 1))),
                                      np.vstack((
                                          f2e_is_dir_part_mat[:, [2, 0]],
                                          f2e_is_dir_part_mat[:, [0, 1]],
                                          f2e_is_dir_part_mat[:, [1, 2]]
                                      )))

            f2e2_new_mat = n_edges + \
                np.hstack([(ml.repmat(ind_shift_edge_vec, 1, 3) +
                          np.kron(np.array([1, 0, 2]) * n_shrinked_faces, np.ones(shape=(1, n_shrinked_faces)))).T,
                          3 * n_shrinked_faces - n_verts + dir_mat * n_new_verts +
                          np.vstack([
                              np.column_stack([ind_vf13_vec, ind_vf12_vec]),
                              np.column_stack([ind_vf12_vec, ind_vf23_vec]),
                              np.column_stack([ind_vf23_vec, ind_vf13_vec])
                          ])
                          ])

            f2e2_is_dir_new_mat = np.array(np.hstack([(np.kron(np.array([1, 0, 0]),
                                                               np.ones(shape=(1, n_shrinked_faces)))).T,
                                                      dir_mat]), dtype=bool)

            # Build the third group of new faces that correspond to edges
            # that have only 1 partitioned face of 2 adjacent faces

            # - this identifies edges in e_mat
            ind_e_broken_but_kept_vec = ind_e_kept_vec[is_e2part_vec & is_e_kept_vec]

            # but we still need to find indices of new vertices in the middles of these edges
            ind_v_mid_e_broken_nut_kept_vec = ind_e_part_vec[is_e2part_vec & is_e_kept_vec]

            # faces have zero volume
            f3_new_mat = np.column_stack((ind_v_mid_e_broken_nut_kept_vec, e_mat[ind_e_broken_but_kept_vec]))

            ind_12f2e3_vec = ind_v_mid_e_broken_nut_kept_vec - n_verts + 3 * n_shrinked_faces + n_edges
            ind_23f2e3_vec = ind_12f2e3_vec + n_new_verts

            f2e3_new_mat = np.column_stack((ind_12f2e3_vec, ind_23f2e3_vec, ind_e_broken_but_kept_vec))
            f2e3_is_dir_new_mat = np.ones(shape=np.shape(f2e3_new_mat), dtype=bool)

            # Optionally adjust vertices
            if is_adjust_func_spec:
                v_new_mat = f_vert_adjust_func(v_new_mat)

            # Update edges, faces, vertices and f2e_map
            v_mat = np.vstack((v_mat, v_new_mat))
            e_mat = np.vstack((e_mat, e1_new_mat, e2_new_mat))
            f_mat = np.vstack((f_mat, f1_new_mat, f2_new_mat, f3_new_mat))
            f2e_mat = np.array(np.vstack((f2e_mat, f2e1_new_mat, f2e2_new_mat, f2e3_new_mat)), dtype=np.int32)
            f2e_is_dir_mat = np.vstack((f2e_is_dir_mat, f2e1_is_dir_new_mat, f2e2_is_dir_new_mat, f2e3_is_dir_new_mat))

            # Update edge length vec
            d_mat = v_mat[e_mat[n_edges:, 0]] - v_mat[e_mat[n_edges:, 1]]
            e_length_vec = np.concatenate((e_length_vec, np.sqrt(np.sum(d_mat * d_mat, 1))))

            # Update number of entities
            n_verts = np.size(v_mat, 0)
            n_edges = np.size(e_mat, 0)
            n_faces = np.size(f_mat, 0)

            # Collect stats
            if is_stat_collected:
                n_vert_vec = np.append(n_vert_vec, n_verts)
                n_edge_vec = np.append(n_edge_vec, n_edges)
                n_face_vec = np.append(n_face_vec, n_faces)

            i_step += 1
        else:
            break
    if is_stat_collected:
        s_stats = dict()
        s_stats['n_steps'] = i_step - 1
        s_stats['n_vert_vec'] = n_vert_vec
        s_stats['n_face_vec'] = n_face_vec
        s_stats['n_edge_vec'] = n_edge_vec
        s_stats['n_edges_to_shrink_vec'] = n_edges_to_shrink_vec
        s_stats['max_edge_length_vec'] = max_edge_length_vec
        return v_mat, f_mat, s_stats, e_mat, f2e_mat, f2e_is_dir_mat
    else:
        return v_mat, f_mat

        
def sphere_tri(depth: int) -> Tuple[np.ndarray, np.ndarray]:
    def normvert(x: np.ndarray) -> np.ndarray:
        return x / ml.repmat(np.sqrt(np.sum(x * x, 1)).reshape(-1, 1), 1, 3)

    if not (np.isscalar(depth) and is_numeric(np.array(depth))
            and 0 <= depth == np.fix(depth)):
        throw_error('wrong_input',
                    'depth is expected to be a non-negative integer scalar')
    v_mat, f_mat = icosahedron()
    v_mat, f_mat = shrink_face_tri(v_mat, f_mat, 0, depth, normvert)
    return v_mat, f_mat


def is_face(f_mat: np.ndarray, f_to_check_mat: np.ndarray) -> np.ndarray:
    def get_edge_attachements(
            inp_f_num_list: List[np.ndarray],
            inp_e_mat: np.ndarray, inp_check_e_mat: np.ndarray) -> List[np.ndarray]:
        def as_void(arr):
            arr = np.ascontiguousarray(arr)
            return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

        inp_check_e_mat = np.sort(inp_check_e_mat, axis=1)
        void_e_mat, void_check_e_mat = map(as_void, (inp_e_mat, inp_check_e_mat))
        is_edge_vec, ind_edge_vec = is_member(void_check_e_mat, void_e_mat)
        out_f_num_list = list()
        i_edge = 0
        for i_check_edge in range(inp_check_e_mat.shape[0]):
            if is_edge_vec[i_check_edge]:
                out_f_num_vec = inp_f_num_list[ind_edge_vec[i_edge]]
                i_edge += 1
            else:
                out_f_num_vec = np.empty((0,), dtype=np.int32)
            out_f_num_list.append(out_f_num_vec)
        return out_f_num_list

    f_mat, e_mat, ind_f_vec, ind_shift_vec = get_face_edge_info(f_mat)
    if ind_shift_vec.size == 1:
        f_num_list = [ind_f_vec]
    else:
        f_num_list = np.split(ind_f_vec, np.cumsum(ind_shift_vec[:-1]))

    f12_num_list = get_edge_attachements(f_num_list, e_mat, f_to_check_mat[:, [0, 1]])
    f23_num_list = get_edge_attachements(f_num_list, e_mat, f_to_check_mat[:, [1, 2]])
    f13_num_list = get_edge_attachements(f_num_list, e_mat, f_to_check_mat[:, [0, 2]])

    def fnum_list_to_mat(inp_f_num_list: List[np.ndarray], def_vec: np.array) -> np.ndarray:
        n_elems = len(inp_f_num_list)
        l_vec = np.array([f_num_vec.size for f_num_vec in inp_f_num_list])
        f_num_mat = np.tile(def_vec, (n_elems, 1))
        is_one_elem_vec = l_vec == 1
        if np.any(is_one_elem_vec):
            f_num_mat[is_one_elem_vec, 0] = \
                np.hstack(tuple([f_num_vec for f_num_vec, is_elem in
                                 zip(inp_f_num_list, list(is_one_elem_vec)) if is_elem]))
        is_two_elem_vec = l_vec == 2
        if np.any(is_two_elem_vec):
            f_num_mat[is_two_elem_vec, :] = \
                np.vstack(tuple([f_num_vec for f_num_vec, is_elem in
                                 zip(inp_f_num_list, list(is_two_elem_vec)) if is_elem]))
        return f_num_mat

    f12_num_mat = fnum_list_to_mat(f12_num_list, np.array([-121, -122]))
    f23_num_mat = fnum_list_to_mat(f23_num_list, np.array([-231, -232]))
    f13_num_mat = fnum_list_to_mat(f13_num_list, np.array([-131, -132]))

    def intersect_mat(one_mat: np.ndarray, two_mat: np.ndarray, three_mat: np.ndarray) -> np.ndarray:
        def flip_or_not(inp_mat: np.ndarray, is_flip: bool) -> np.ndarray:
            if is_flip:
                return np.fliplr(inp_mat)
            else:
                return inp_mat

        comb_mat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        n_combs = comb_mat.shape[0]
        is_pos_res_vec = np.zeros((one_mat.shape[0],), dtype=bool)
        for i_comb in range(n_combs):
            is_flip_vec = comb_mat[i_comb]
            flip_one_mat = flip_or_not(one_mat, is_flip_vec[0])
            flip_two_mat = flip_or_not(two_mat, is_flip_vec[1])
            flip_three_mat = flip_or_not(three_mat, is_flip_vec[2])
            is_pos_res_vec = \
                np.logical_or(
                    is_pos_res_vec, np.any(np.logical_and(
                        flip_one_mat == flip_two_mat, flip_two_mat == flip_three_mat), axis=1).flatten())
        return is_pos_res_vec

    return intersect_mat(f12_num_mat, f23_num_mat, f13_num_mat)


def is_tri_equal(v1_mat: np.ndarray, f1_mat: np.ndarray,
                 v2_mat: np.ndarray, f2_mat: np.ndarray, max_tol: float) -> Tuple[bool, str]:

    n1_verts = v1_mat.shape[0]
    n2_verts = v2_mat.shape[0]
    is_pos = n1_verts == n2_verts

    if is_pos:
        n1_faces = f1_mat.shape[0]
        n2_faces = f2_mat.shape[0]
        is_pos = n1_faces == n2_faces
        if is_pos:
            v1_mat, _, ind_f1_vec = sort_rows_tol(v1_mat, max_tol)
            v2_mat, _, ind_f2_vec = sort_rows_tol(v2_mat, max_tol)

            f1_mat = ind_f1_vec[f1_mat]
            f2_mat = ind_f2_vec[f2_mat]
            real_tol = np.max(np.max(np.abs(v1_mat-v2_mat)))
            is_pos = real_tol <= max_tol

            if is_pos:
                n_f1_unique = np.size(np.unique(f1_mat, axis=0), 0)
                n_f2_unique = np.size(np.unique(f2_mat, axis=0), 0)
                is_pos = n_f1_unique == n_f2_unique
                if is_pos:
                    is_pos = np.all(is_face(f2_mat, f1_mat))
                    if is_pos:
                        report_str = ''
                    else:
                        report_str = 'faces are different'
                else:
                    report_str = 'numbers of unique faces are different'
            else:
                report_str = 'vertices are different, real tol={}, exp tol={}'.format(real_tol, max_tol)
        else:
            report_str = 'number of faces is different'
    else:
        report_str = 'numbers of vertices are different'
    return is_pos, report_str

    
def sphere_tri_ext(n_dim: int, n_points: int, return_f_grid: bool = False)\
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    def spherebndr_2d(n_points: int, return_f_vec: bool = False) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        bp_mat = circle_part(n_points)
        if return_f_vec:
            f_vec = np.ndarray(shape=(n_points, 2),
                               buffer=np.array([np.arange(1, n_points),
                                                np.arange(2, n_points + 1)]), dtype=int).T
            f_vec[n_points - 1, 1] = 1
            return bp_mat, f_vec
        return bp_mat

    def spherebndr_3d(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        sphere_triang_num = calc_depth(n_points)
        bp_mat, f_mat = sphere_tri(sphere_triang_num)
        return bp_mat, f_mat

    def calc_depth(n_points: int) -> int:
        # Initial icosaeder parameters:
        __VERTICES_NUM = 12
        __FACES_NUM = 20
        __EDGES_NUM = 30
        vert_num = __VERTICES_NUM
        face_num = __FACES_NUM
        edge_num = __EDGES_NUM
        #
        cur_depth = 0
        is_stop = False
        while not is_stop:
            cur_depth = cur_depth + 1
            vert_num = vert_num + edge_num
            edge_num = 2 * edge_num + 3 * face_num
            face_num = 4 * face_num
            is_stop = vert_num >= n_points
        triang_depth = cur_depth
        return triang_depth

    if not (np.isscalar(n_points) and is_numeric(np.array(n_points))
            and 0 < n_points == np.fix(n_points)):
        throw_error('wrong_input',
                    'n_points is expected to be a positive integer scalar number')
    if n_dim == 2:
        if return_f_grid:
            v_grid_mat, f_grid_mat = spherebndr_2d(n_points, return_f_grid)
        else:
            v_grid_mat = spherebndr_2d(n_points)
            (v_grid_mat == 0).choose(v_grid_mat, np.finfo(float).eps)
            return v_grid_mat
    else:
        v_grid_mat, f_grid_mat = spherebndr_3d(n_points)
    (v_grid_mat == 0).choose(v_grid_mat, np.finfo(float).eps)
    return v_grid_mat, f_grid_mat
