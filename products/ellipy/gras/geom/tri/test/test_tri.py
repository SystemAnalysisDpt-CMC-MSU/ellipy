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
        def check(depth: int):
            def check_vert(v: np.ndarray):
                norm_vec = np.sqrt(np.sum(v * v, axis=1))
                is_pos = np.max(np.abs(norm_vec - 1)) <= __MAX_TOL
                assert is_pos, 'not all vertices are on the unit sphere'

            def check_regress(v_1: np.ndarray, f_1: np.ndarray, curr_depth: int):
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
            # noinspection PyUnresolvedReferences
            cf0 = hull_0.simplices
            # noinspection PyUnresolvedReferences
            vol0 = hull_0.volume
            hull_1 = ConvexHull(v1)
            # noinspection PyUnresolvedReferences
            cf1 = hull_1.simplices
            # noinspection PyUnresolvedReferences
            vol1 = hull_1.volume
            assert vol1 > vol0
            assert vol1 < np.pi * 4 / 3
            assert cf0.shape[0] * 4 == cf1.shape[0]

        __MAX_TOL = 1e-13
        __MAX_DEPTH = 4
        for cur_depth in np.arange(1, __MAX_DEPTH + 1):
            check(cur_depth)

    def test_sphere_tri_ext(self):
        __N_POINTS = 500
        __RIGHT_POINTS_3D = 642
        dim = 2
        v_mat = sphere_tri_ext(dim, __N_POINTS)
        assert v_mat.shape[0] == __N_POINTS
        dim = 3
        v_mat, _ = sphere_tri_ext(dim, __N_POINTS)
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

    def test_is_face(self):
        f_mat = self.__TRI3_FACE
        f_to_check_mat = np.vstack((np.array([[0, 4, 3]]), f_mat, np.array([[1, 4, 3]])))
        is_face_there_vec = is_face(f_mat, f_to_check_mat)
        assert np.array_equal(is_face_there_vec, np.array([False, True, True, True, False]))

    def test_shrink_face_tri(self):
        __MAX_DIST = 0.5
        __N_DATA_SETS = 2

        def as_void(arr):
            arr = np.ascontiguousarray(arr)
            return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

        def check_step_wise(v_inp_mat: np.ndarray, f_inp_mat: np.ndarray, max_tol: float, *args):
            __MAX_TOL = 1e-14
            v_res_mat, f_res_mat, s_stat = self.aux_shrink_face_tri(v_inp_mat, f_inp_mat, max_tol, *args)
            n_steps = s_stat['n_steps']
            if n_steps > 1:
                v_1mat, f_1mat, _ = self.aux_shrink_face_tri(v_inp_mat, f_inp_mat, max_tol, n_steps - 1)
                v2_mat, f2_mat, _ = self.aux_shrink_face_tri(v_1mat, f_1mat, max_tol, 1)
                is_pos, report_str = is_tri_equal(v_res_mat, f_res_mat, v2_mat, f2_mat, __MAX_TOL)
                assert is_pos, report_str

        def shrink(v_0_mat: np.ndarray, f_0_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # shrink faces
            v_1_mat, f_1_mat, s1_stat = self.aux_shrink_face_tri(v_0_mat, f_0_mat, __MAX_DIST)
            # Perform additional checks
            v_2_mat, f_2_mat, s2_stat = self.aux_shrink_face_tri(v_0_mat, f_0_mat, __MAX_DIST,
                                                                 s1_stat['n_steps'])
            assert np.array_equal(v_1_mat, v_2_mat)
            assert np.array_equal(f_1_mat, f_2_mat)
            assert s1_stat.keys() == s2_stat.keys()
            for key in s1_stat.keys():
                np.array_equal(s1_stat[key], s2_stat[key])
            check_step_wise(v_0_mat, f_0_mat, 0, 3)
            check_step_wise(v_0_mat, f_0_mat, __MAX_DIST)
            return v_1_mat, f_1_mat

        for i_data_set in np.r_[__N_DATA_SETS:0:-1]:
            loaded_info = sio.loadmat(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inp%s.mat' % i_data_set))
            v0_mat = loaded_info['v0']
            f0_mat = loaded_info['f0']
            v1_mat, f1_mat = shrink(v0_mat, f0_mat - 1)
            # check that no vertices is deleted
            void_v0_mat, void_v1_mat = map(as_void, (v0_mat, v1_mat))
            assert np.all(np.in1d(void_v0_mat, void_v1_mat))
            # check that all edges are short enough
            ind_2_check = np.array([[0, 1], [0, 2], [1, 2]])
            pos_edges = np.sort(np.reshape(f1_mat[:, ind_2_check], newshape=(-1, 2)))
            e1_mat = np.unique(pos_edges, axis=0)
            d_mat = v1_mat[e1_mat[:, 0], :] - v1_mat[e1_mat[:, 1], :]
            max_edge_length = np.max(np.sqrt(np.sum(d_mat * d_mat, axis=1)))
            assert max_edge_length <= __MAX_DIST
            # regression test
            loaded_info = sio.loadmat(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out%s.mat' % i_data_set))
            s_out_v0 = v1_mat
            s_out_f0 = f1_mat
            se_out_v0 = loaded_info['v0']
            se_out_f0 = loaded_info['f0']
            #
            is_ok, rep_str = is_tri_equal(se_out_v0, se_out_f0 - 1, s_out_v0, s_out_f0, 0)
            assert is_ok, rep_str
