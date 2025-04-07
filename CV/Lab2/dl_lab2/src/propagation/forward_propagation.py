import numpy as np
from logger import logger
from abc import ABC, abstractmethod
from src.params.weights import ParamsContainer
from dataclasses import dataclass

# Data orientated class to store outputs of Forward Prpagation stepp
@dataclass
class ForwardPathContainer:
    Z1: np.ndarray
    A1: np.ndarray
    Z2: np.ndarray
    A2: np.ndarray

# Introducing Interface of Forward Propagation class
class ForwardPropagationI(ABC):
    @staticmethod
    @abstractmethod
    def forward(X: np.ndarray, params: dict[np.ndarray]) -> ForwardPathContainer:
        ...
        
        
# Implementation of concrete class which will perform Forwar Propagation
class ForwardPropagation(ForwardPropagationI):
    @staticmethod
    def forward(X: np.ndarray, params: ParamsContainer) -> ForwardPathContainer:
        try:
            Z1 = params.W1 @ X + params.b1
            A1 = np.tanh(Z1)
            Z2 = params.W2 @ A1 + params.b2
            A2 = (1 / (1 + np.e**(-Z2)))
            logger.info('Forward path done succesfully')
            
            return A2, ForwardPathContainer(Z1=Z1,
                                            A1=A1,
                                            Z2=Z2,
                                            A2=A2)
            
        except Exception as e:
            logger.exception(e)
            raise e    
        
    def __repr__(self) -> str:
        return f"ForwardPropagation()"
    
if __name__ == '__main__':
    prop = ForwardPropagation()
    print(prop)
    
        
