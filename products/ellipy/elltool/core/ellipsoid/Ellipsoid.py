from ellipy.elltool.core.aellipsoid.AEllipsoid import *
# from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable
import numpy as np


class Ellipsoid(AEllipsoid):
    def __init__(self):
        AEllipsoid.__init__(self)

    @property
    def _center_vec(self) -> np.matrix:
        return self.__center_vec

    @_center_vec.setter
    def _center_vec(self, center_vec: np.matrix):
        self.__center_vec = center_vec

    def to_dict(self, is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[Dict[str, np.matrix], Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.matrix], np.matrix]]]:
        pass
