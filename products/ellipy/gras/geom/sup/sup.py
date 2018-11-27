import numpy as np


def sup_2_boundary_2(dir_mat, sup_vec):
    n_dirs = (np.shape(dir_mat)[0])
    x_bound_mat = np.zeros((2, n_dirs))
    for i_dir in range(n_dirs - 1):
        x_bound_mat[:, i_dir:i_dir+1] = np.dot(np.linalg.pinv(dir_mat[i_dir:i_dir + 2]),
                                               sup_vec[i_dir:i_dir + 2].reshape(-1, 1))

    x_bound_mat[:, -1:] = np.dot(np.linalg.pinv(np.array((dir_mat[n_dirs - 1, :], dir_mat[0, :]))),
                                 np.array(([sup_vec[n_dirs - 1]], [sup_vec[0]])).reshape(-1, 1))
    x_bound_mat = x_bound_mat.transpose()
    return x_bound_mat


def sup_2_boundary_3(dir_mat, sup_vec, face_mat):
    n_faces = (np.shape(face_mat)[0])
    x_bound_mat = np.zeros((3, n_faces))

    for i_face in range(n_faces):
        x_bound_mat[:, i_face:i_face + 1] = np.dot(np.linalg.pinv(dir_mat[face_mat[i_face:i_face + 1, :], :]),
                                                   sup_vec[face_mat[i_face:i_face + 1, :]].reshape(-1, 1))
    x_bound_mat = x_bound_mat.transpose()
    return x_bound_mat