from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable
import numpy as np


class ABasicEllipsoid(ABC):
    def __init__(self):
        pass

    def _before_get_abs_tol(self) -> None:
        pass

    @property
    def _abs_tol(self) -> float:
        self._before_get_abs_tol()
        return self.__abs_tol

    @_abs_tol.setter
    def _abs_tol(self, abs_tol: float):
        self.__abs_tol = abs_tol

    def _before_get_rel_tol(self) -> None:
        pass

    @property
    def _rel_tol(self) -> float:
        self._before_get_rel_tol()
        return self.__rel_tol

    @_rel_tol.setter
    def _rel_tol(self, rel_tol: float):
        self.__rel_tol = rel_tol

    @abstractmethod
    def dimension(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def to_dict(self, is_prop_included: bool = False, abs_tol: float = None) -> \
            Tuple[Dict[str, np.matrix], Dict[str, str], Dict[str, str],
                  Dict[str, Callable[[np.matrix], np.matrix]]]:
        pass

    def _is_equal_internal(self, other, is_prop_included: bool) -> Tuple[bool, str]:
        pass

    def __eq__(self, other) -> bool:
        is_eq, _ = ABasicEllipsoid._is_equal_internal(self, other, False)
        return is_eq
