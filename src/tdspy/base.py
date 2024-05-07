"""
Base for objects: RDDE, NDDE, DDAE
----------------------------------
"""

from abc import ABC, abstractmethod

import numpy.typing as npt

class TDSBase(ABC):

    @property
    @abstractmethod
    def mA(self) -> int:
        ...
    
    @property
    @abstractmethod
    def hA(self) -> npt.NDArray:
        ...

    @property
    @abstractmethod
    def A(self) -> npt.NDArray:
        ...

    @property
    @abstractmethod
    def E(self) -> npt.NDArray:
        ...

    @property
    @abstractmethod
    def n(self) -> int:
        """ state size """
        ...
    
    @property
    @abstractmethod
    def p2(self) -> int:
        """ input size """
        ...
