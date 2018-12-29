from scipy.spatial import ConvexHull
from ellipy.gen.common.common import throw_error
import numpy as np
import numpy.matlib


def sup_2_boundary_2(dir_mat: np.ndarray, sup_vec: np.ndarray) -> np.ndarray:
    n_dirs = np.shape(dir_mat)[0]
    x_bound_mat = np.zeros((2, n_dirs), dtype=np.float64)
    for i_dir in range(n_dirs - 1):
        x_bound_mat[:, i_dir] = np.linalg.lstsq(dir_mat[i_dir:i_dir + 2],
                                                sup_vec[i_dir:i_dir + 2], -1)[0]

    x_bound_mat[:, -1] = np.linalg.lstsq(np.vstack((dir_mat[n_dirs - 1, :], dir_mat[0, :])),
                                         np.hstack((sup_vec[n_dirs - 1], sup_vec[0])), -1)[0]
    x_bound_mat = x_bound_mat.T
    return x_bound_mat


def sup_2_boundary_3(dir_mat: np.ndarray, sup_vec: np.ndarray, face_mat: np.ndarray) -> np.ndarray:
    n_faces = face_mat.shape[0]
    x_bound_mat = np.zeros((3, n_faces), dtype=np.float64)

    for i_face in range(n_faces):
        x_bound_mat[:, i_face] = np.linalg.lstsq(dir_mat[face_mat[i_face, :], :],
                                                 sup_vec[face_mat[i_face, :]], -1)[0]
    x_bound_mat = x_bound_mat.T
    return x_bound_mat


def sup_geom_diff_2d(rho1_vec: np.ndarray, rho2_vec: np.ndarray, l_mat: np.ndarray) -> np.ndarray:
    if l_mat.ndim != 2:
        throw_error('wrongInput:l_mat',
                    'l_mat must be a matrix')

    if rho1_vec.size != max(rho1_vec.shape):
        throw_error('wrongInput:rho1_vec',
                    'rho1_vec must be a vector')

    if rho2_vec.size != max(rho2_vec.shape):
        throw_error('wrongInput:rho2_vec',
                    'rho2_vec must be a vector')

    if rho1_vec.ndim == 0:
        rho1_vec = np.expand_dims(rho1_vec, axis=0)
    else:
        rho1_vec = np.squeeze(rho1_vec)

    if rho2_vec.ndim == 0:
        rho2_vec = np.expand_dims(rho2_vec, axis=0)
    else:
        rho2_vec = np.squeeze(rho2_vec)

    if rho1_vec.shape[0] != rho2_vec.shape[0]:
        throw_error('wrongInput:rho1_vec,rho2_vec',
                    'rho1_vec, rho2_vec must have the same length')

    if rho1_vec.shape[0] != l_mat.shape[1]:
        throw_error('wrongInput:rho1_vec,l_mat',
                    'The number of columns of l_mat must equal the length of rho1_vec')

    n_dirs = l_mat.shape[1]
    n_dims = l_mat.shape[0]

    if n_dims != 2:
        throw_error('wrongInput:l_mat', 'Only 2-dimensional sets are supported')

    rho_diff_vec = rho1_vec - rho2_vec

    if np.any(rho_diff_vec <= 0):
        throw_error('wrongInput:rho_diff_vec',
                    'Geometric difference is expected to have a non-empty interior')

    s_mat = l_mat / np.matlib.repmat(rho_diff_vec, n_dims, 1)
    # noinspection PyUnresolvedReferences
    ind_mat = ConvexHull(s_mat.T).simplices

    ind_face_length_vec = ind_mat[:, 1] - ind_mat[:, 0]
    is_neg_vec = ind_face_length_vec < -1
    ind_face_length_vec[is_neg_vec] = n_dirs + ind_face_length_vec[is_neg_vec]
    ind_face_vec = np.nonzero(ind_face_length_vec > 1)[0]
    n_faces = len(ind_face_vec)
    s_norm_vec = np.ones(n_dirs, dtype=np.float64)

    for i_face in range(n_faces):
        ind_face = ind_face_vec[i_face]
        ind_left = ind_mat[ind_face, 0]
        ind_right = ind_mat[ind_face, 1]

        b_vec = np.expand_dims(s_mat[:, ind_left], axis=0).T
        a_first_vec = b_vec - np.expand_dims(s_mat[:, ind_right], axis=0).T

        n_change_dirs = ind_face_length_vec[ind_face] - 1
        ind_dir = ind_left

        for i_dir in range(0, n_change_dirs):
            ind_dir = ind_dir + 1

            if ind_dir > n_dirs:
                ind_dir = 0

            a_mat = np.concatenate((a_first_vec, np.expand_dims(s_mat[:, ind_dir], axis=0).T), axis=1)
            x_vec = np.linalg.lstsq(a_mat, b_vec, -1)[0]
            s_val = x_vec[1]

            if s_val < 0:
                throw_error('wrongState:s_val', 's_val cannot be negative')

            s_norm_vec[ind_dir] = s_val

    rho_diff_vec = rho_diff_vec / s_norm_vec

    return rho_diff_vec


def sup_geom_diff_3d(rho1_vec: np.ndarray, rho2_vec: np.ndarray, l_mat: np.ndarray) -> np.ndarray:
    __ABS_TOL = 1e-5

    if l_mat.ndim != 2:
        throw_error('wrongInput:l_mat',
                    'l_mat must be a matrix')

    if rho1_vec.size != max(rho1_vec.shape):
        throw_error('wrongInput:rho1_vec',
                    'rho1_vec must be a vector')

    if rho2_vec.size != max(rho2_vec.shape):
        throw_error('wrongInput:rho2_vec',
                    'rho2_vec must be a vector')

    if rho1_vec.ndim == 0:
        rho1_vec = np.expand_dims(rho1_vec, axis=0)
    else:
        rho1_vec = np.squeeze(rho1_vec)

    if rho2_vec.ndim == 0:
        rho2_vec = np.expand_dims(rho2_vec, axis=0)
    else:
        rho2_vec = np.squeeze(rho2_vec)

    if rho1_vec.shape[0] != rho2_vec.shape[0]:
        throw_error('wrongInput:rho1_vec, rho2_vec',
                    'rho1_vec, rho2_vec must have the same length')

    if rho1_vec.shape[0] != l_mat.shape[1]:
        throw_error('wrongInput:rho1_vec, l_mat',
                    'The number of columns of l_mat must equal the length of rho1_vec')

    n_dims = l_mat.shape[0]
    if n_dims != 3:
        throw_error('wrongInput:l_mat', 'Only 2-dimensional sets are supported')

    rho_diff_vec = rho1_vec - rho2_vec

    if np.any(rho_diff_vec <= 0):
        throw_error('wrongInput:rho_diff_vec',
                    'Geometric difference is expected to have a non-empty interior')

    s_mat = l_mat / np.matlib.repmat(rho_diff_vec, n_dims, 1)
    # noinspection PyUnresolvedReferences
    f_s_mat = ConvexHull(s_mat.T).simplices
    dist_vec = np.zeros(l_mat.shape[1])

    for i_dist in range(l_mat.shape[1]):
        for j_tri in range(f_s_mat.shape[0]):
            tri_mat = s_mat[:, f_s_mat[j_tri, :]]
            x1_vec = tri_mat[:, 1] - tri_mat[:, 0]
            x2_vec = tri_mat[:, 2] - tri_mat[:, 0]
            norm1_vec = np.cross(s_mat[:, i_dist], x2_vec)
            det_temp = np.dot(x1_vec, norm1_vec)
            if np.abs(det_temp) > __ABS_TOL:
                inv_det = 1 / det_temp
                s_point = -tri_mat[:, 0]
                u_dist = inv_det * np.dot(s_point, norm1_vec)
                if u_dist >= -__ABS_TOL:
                    norm2_vec = np.cross(s_point, x1_vec)
                    v_dist = inv_det * np.dot(s_mat[:, i_dist], norm2_vec)
                    if v_dist >= -__ABS_TOL and u_dist + v_dist <= 1 + __ABS_TOL:
                        t_dist = inv_det * np.dot(x2_vec, norm2_vec)
                        if t_dist > 0:
                            dist_vec[i_dist] = t_dist

    rho_diff_vec = rho_diff_vec / dist_vec

    return rho_diff_vec
