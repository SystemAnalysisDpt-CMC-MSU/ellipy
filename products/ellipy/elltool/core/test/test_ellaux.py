from ellipy.elltool.core.ellipsoid.Ellipsoid import *


class EllAuxTestCase:
    __ABS_TOL = 1e-8

    # noinspection PyMethodMayBeStatic
    def ellipsoid(self, *args, **kwargs):
        return Ellipsoid(*args, **kwargs)
