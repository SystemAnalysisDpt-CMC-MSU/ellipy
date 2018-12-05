from ellipy.elltool.core.ellipsoid.Ellipsoid import *
from ellipy.elltool.core.hyperplane.Hyperplane import *


class TestEllipsoidTestCase:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def hyperplane(self, *args, **kwargs):
        return Hyperplane(*args, **kwargs)
