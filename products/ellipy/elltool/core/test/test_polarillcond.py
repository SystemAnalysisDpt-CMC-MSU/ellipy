from ellipy.elltool.core.ellipsoid.Ellipsoid import *


class PolarEllipsoidTest(AEllipsoid):
    pass


class TestPolarIllCondTC:
    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)
