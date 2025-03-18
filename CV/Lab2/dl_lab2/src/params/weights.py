import numpy as np
from logger import logger
from utils import type_check
from abc import ABC, abstractmethod
from dataclasses import dataclass

# IMplementation of data orientated class which will contain all required params
@dataclass
class ParamsContainer:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    
    def __sub__(self, other):
        return ParamsContainer(self.W1 - other[0],
                               self.b1 - other[1],
                               self.W2 - other[2],
                               self.b2 - other[3])

# Introducing interface of the class which will initialize weights base on input hidden layres size
class WeightInitializerI(ABC):
    @staticmethod
    @abstractmethod
    def weights(input_size: int, hidden_size: int, output_size: int) -> ParamsContainer:
        ...
        
class RandomWeightInitializer(WeightInitializerI):
    @staticmethod
    def weights(input_size: int, output_size: int, hidden_size: int = 4) -> ParamsContainer:
        for sample in (input_size, hidden_size, output_size):
            type_check(sample, int)
            
        params = ParamsContainer(W1=np.random.rand(hidden_size, input_size),
                                 b1= np.zeros((hidden_size, 1)),
                                 W2=np.random.rand(output_size, hidden_size),
                                 b2=np.zeros((output_size, 1))
                                 )
        logger.info('Weight has been initialized')
        return params
                
if __name__ == '__main__':
    weights_initializer = RandomWeightInitializer()
    res = weights_initializer.weights(10,10,10)