class TestGeom:
    # Make it for circle_part!!!
    def test_sphere_part(self):
        CALC_PRECISION = 1e-14
        def test_for_n_points(n_points: int):
            pMat = gras.geom.spherepart(n_points);
            normVec = realsqrt(sum(pMat. * pMat, 2));
            mlunitext.assert (size(pMat, 1) == n_points);
            mlunitext.assert (size(pMat, 2) == 3);
            mlunitext.assert (all(abs(normVec - 1) < CALC_PRECISION))

        num_points_vec = [1, 2, 3, 20, 21, 22, 41, 42, 43, 100]
        arrayfun(test_for_n_points, num_points_vec)
