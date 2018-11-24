from ellipy.gras.geom.tri.tri import *
from ellipy.gen.common.common import *
import numpy as np
from numpy import matlib as ml
import scipy.io as sio
import os
from scipy.spatial import ConvexHull


class TestTri:
    __TRI1_VERT = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    __TRI1_FACE = np.array([0, 1, 2])

    __TRI2_VERT = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1]
    ])
    __TRI2_FACE = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 1, 3],
        [1, 2, 3]
    ])

    __TRI3_VERT = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-0.5, 0, 0],
        [0, -0.5, 0]
    ])
    __TRI3_FACE = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 4, 1]
    ])
    __TRI3_EDGE = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 4],
        [2, 3]
    ])
    __TRI3_F2E = np.array([
        [0, 4, 1],
        [1, 6, 2],
        [3, 5, 0]
    ])
    __TRI3_F2E_DIR = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1]
    ], dtype=bool)

    __TRI31_VERT = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-0.5, 0, 0]
    ])
    __TRI31_FACE = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    @staticmethod
    def aux_shrink_face_tri(v_mat, f_mat, *args):
        v_mat, f_mat, s_stats, e_mat, f2e_mat, _ = shrink_face_tri(v_mat, f_mat, is_stat_collected=True, *args)
        n_edges = np.size(e_mat, 0)
        assert n_edges == len(np.unique(f2e_mat))
        return v_mat, f_mat, s_stats

    def test_shrink_face_tri_one_face(self):
        v_mat = self.__TRI1_VERT
        f_mat = self.__TRI1_FACE
        v1_mat, f1_mat, _ = self.aux_shrink_face_tri(v_mat, f_mat, 0, 1)
        v1_exp_mat = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5]
        ])
        f1_exp_mat = np.array([
            [3, 5, 4],
            [3, 4, 0],
            [5, 3, 1],
            [4, 5, 2]
        ])
        assert np.array_equal(v1_mat, v1_exp_mat)
        assert np.array_equal(f1_mat, f1_exp_mat)
        f_mat = np.array([2, 1, 0])
        _, *_ = self.aux_shrink_face_tri(v_mat, f_mat, 0, 1)
        f_mat = np.array([1, 2, 0])
        _, *_ = self.aux_shrink_face_tri(v_mat, f_mat, 0, 1)
        f_mat = np.array([2, 0, 1])
        _, *_ = self.aux_shrink_face_tri(v_mat, f_mat, 0, 1)
        _, *_ = self.aux_shrink_face_tri(v_mat, f_mat, 0, 4)

    def test_shrink_face_tri_3_faces(self):
        v_mat = self.__TRI2_VERT
        f_mat = self.__TRI2_FACE
        _, *_ = self.aux_shrink_face_tri(v_mat, f_mat, 0, 2)

    def test_shrink_face_tri_2_face_1_part(self):
        v_mat = self.__TRI31_VERT
        f_mat = self.__TRI31_FACE
        _, *_ = self.aux_shrink_face_tri(v_mat, f_mat, np.sqrt(2) - 0.001, 1)

    def test_shrink_face_tri_3_face_1_part(self):
        v_mat = self.__TRI3_VERT
        f_mat = self.__TRI3_FACE
        f_vert_adjust_func: Callable[[np.ndarray], np.ndarray] = lambda x: x + ml.repmat(np.array([0, 0, 0.2]),
                                                                                         np.size(x, 0), 1)
        v1_mat, f1_mat, _ = self.aux_shrink_face_tri(v_mat, f_mat, np.sqrt(2) - 0.001, 1, f_vert_adjust_func)
        is_face_there_vec = is_face(f1_mat, np.array([[0, 5, 1], [0, 6, 2]]))
        assert np.all(is_face_there_vec)
        _, *_ = self.aux_shrink_face_tri(v1_mat, f1_mat, 0, 3, f_vert_adjust_func)

    def test_map_face_2_edge(self):
        f_mat = self.__TRI3_FACE
        e_exp_mat = self.__TRI3_EDGE
        f2e_exp_mat = self.__TRI3_F2E
        f2e_exp_is_dir_mat = self.__TRI3_F2E_DIR
        e_mat, f2e_mat, f2e_is_dir_mat = map_face_2_edge(f_mat)
        assert np.array_equal(e_mat, e_exp_mat)
        assert np.array_equal(f2e_mat, f2e_exp_mat)
        assert np.array_equal(f2e_is_dir_mat, f2e_exp_is_dir_mat)

    def test_sphere_tri(self):
        def check(depth):
            def check_vert(v):
                norm_vec = np.sqrt(np.sum(v * v, axis=1))
                is_pos = np.max(np.abs(norm_vec - 1)) <= __MAX_TOL
                assert is_pos, 'not all vertices are on the unit sphere'

            def check_regress(v_1, f_1, curr_depth):
                loaded_info = sio.loadmat(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_input.mat'))
                cell_mat = loaded_info['cell_mat']
                v_reg_1 = cell_mat[curr_depth - 1, 0]
                f_reg_1 = cell_mat[curr_depth - 1, 1]
                check_vert(v_reg_1)
                v_reg_1 = v_reg_1 / ml.repmat(np.sqrt(np.sum(v_reg_1 * v_reg_1, axis=1).reshape(-1, 1)), 1, 3)
                assert abs_rel_compare(v_reg_1, v_1, __MAX_TOL, None, lambda x: np.abs(x))
                assert np.array_equal(f_reg_1, f_1 + 1)

            v0, f0 = sphere_tri(depth)
            check_regress(v0, f0, depth)
            check_vert(v0)
            #
            v1, f1 = sphere_tri(depth + 1)
            check_regress(v1, f1, depth + 1)
            check_vert(v1)
            hull_0 = ConvexHull(v0)
            cf0 = hull_0.simplices
            vol0 = hull_0.volume
            hull_1 = ConvexHull(v1)
            cf1 = hull_1.simplices
            vol1 = hull_1.volume
            assert vol1 > vol0
            assert vol1 < np.pi * 4 / 3
            assert cf0.shape[0] * 4 == cf1.shape[0]

        __MAX_TOL = 1e-13
        __MAX_DEPTH = 4
        for cur_depth in np.arange(1, __MAX_DEPTH + 1):
            check(cur_depth)

    def test_sphere_tri_ext(self):
        __dim = 2
        __N_POINTS = 500
        __RIGHT_POINTS_3D = 642
        v_mat = sphere_tri_ext(__dim, __N_POINTS)
        assert v_mat.shape[0] == __N_POINTS
        __dim = 3
        v_mat, _ = sphere_tri_ext(__dim, __N_POINTS)
        assert v_mat.shape[0] == __RIGHT_POINTS_3D

    def test_icosahedron(self):
        __MAX_TOL = 1e-13
        loaded_info = sio.loadmat(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_inp_vmat.mat'))
        v_mat = loaded_info['vMat']

        loaded_info = sio.loadmat(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_inp_fmat.mat'))
        f_mat = loaded_info['fMat']

        v_mat_py, f_mat_py = icosahedron()
        assert abs_rel_compare(v_mat_py, v_mat, __MAX_TOL, None, lambda x: np.abs(x))
        assert np.array_equal(f_mat, f_mat_py + 1)

    def test_shrink_face_tri(self):
        pass


    def test_is_face(self):
        v_mat = self.__TRI3_VERT
        f_mat = self.__TRI3_FACE
        f_to_check_mat = np.vstack((np.array([[0, 4, 3]]), f_mat, np.array([[1, 4, 3]])))
        is_face_there_vec = is_face(f_mat, f_to_check_mat)
        assert np.array_equal(is_face_there_vec, np.array([False, True, True, True, False]))
