from ellipy.elltool.core.abasicellipsoid.ABasicEllipsoid import *
from abc import ABC, abstractmethod
# from typing import Union, List, Tuple, Dict, Callable
import numpy as np


class AEllipsoid(ABasicEllipsoid, ABC):
    def __init__(self):
        ABasicEllipsoid.__init__(self)

    @property
    @abstractmethod
    def _center_vec(self) -> np.ndarray:
        pass
