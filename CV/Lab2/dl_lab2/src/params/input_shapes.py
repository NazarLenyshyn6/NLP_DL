import numpy as np
from logger import logger
from abc import ABC, abstractmethod
from typing import Tuple
from utils import type_check

# Introducing interface of the Input shape retrival object
class InputShapeRetrivalI(ABC):
    @staticmethod
    @abstractmethod
    def shape(sample: np.ndarray) -> TypeError | Tuple:
        ...
        
        
# Implementation of concrete obect which retrive shapes of input
class SimpleShapeRetrival(InputShapeRetrivalI):
    @staticmethod
    def shape(sample: np.ndarray) -> TypeError | Tuple:
        type_check(sample, np.ndarray)
        return sample.shape
    
if __name__== '__main__':
    shapes_retrival = SimpleShapeRetrival()
    print(shapes_retrival.shape(np.zeros(10)))