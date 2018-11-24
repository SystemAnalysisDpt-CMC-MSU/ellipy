from ellipy.gras.geom.tri.tri import *
from ellipy.gras.geom.geom import *
import numpy as np
from numpy import matlib as ml


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
            v_0, f_0 = sphere_tri(depth)
        def check_regress(v1, f1, depth):
            [vReg1, fReg1] = gras.geom.tri.test.srebuild3d(depth);
            check_vert(vReg1)
            vReg1 = vReg1 / repmat(realsqrt(sum(vReg1*vReg1, 2)), 1, 3)
            [isPos, reportStr] = gras.geom.tri.istriequal(vReg1, fReg1, v1, f1, MAX_TOL)
            mlunitext.assert_equals(true, isPos, reportStr)
        def check_vert(v):
            normVec = realsqrt(sum(v*v, 2))
            isPos = max(abs(normVec - 1)) <= MAX_TOL
            mlunitext.assert_equals(true, isPos,
                                    'not all vertices are on the unit sphere')
        check_regress(v_0, f_0, depth - 1)
        check_vert(v_0)
        v_1, f_1 = sphere_tri(depth + 1)

        check_regress(v_1, f_1, depth)
        check_vert(v_1)
        [cf0, vol0] = convhull(v_0)
        [cf1, vol1] = convhull(v_1)
        assert vol1 > vol0
        assert vol1 < pi * 4 / 3
        assert cf0.shape[0] * 4 == cf1.shape[0]

        __MAX_TOL = 1e-13
        __MAX_DEPTH = 4
        for curDepth in np.arange(1, __MAX_DEPTH + 1):
            check(curDepth)


    def test_sphere_tri_ext(self):
        __dim = 2
        __N_POINTS = 500
        __RIGHT_POINTS_3D = 642
        v_mat = sphere_tri_ext(__dim, __N_POINTS)
        assert v_mat.shape[0] == __N_POINTS
        __dim = 3
        v_mat = sphere_tri_ext(__dim, __N_POINTS)
        assert v_mat.shape[0] == __RIGHT_POINTS_3D


    def test_shrink_face_tri(self):
        pass


    def test_is_face(self):
        v_mat = self.__TRI3_VERT
        f_mat = self.__TRI3_FACE
        f_to_check_mat = np.vstack((np.array([[0, 4, 3]]), f_mat, np.array([[1, 4, 3]])))
        is_face_there_vec = is_face(f_mat, f_to_check_mat)
        assert np.array_equal(is_face_there_vec, np.array([False, True, True, True, False]))
