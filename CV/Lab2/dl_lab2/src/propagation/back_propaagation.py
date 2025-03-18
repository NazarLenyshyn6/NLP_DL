import numpy as np
from logger import logger
from abc import ABC, abstractmethod
from src.params.weights import ParamsContainer 
from src.propagation.forward_propagation import ForwardPathContainer
from dataclasses import dataclass


# Data Orientated class which will return calculated gradients
@dataclass
class Grads:
    dW1: np.ndarray
    db1: np.ndarray
    dW2: np.ndarray
    db2: np.ndarray
    
    @property
    def values(self):
        return (self.dW1, self.db1, self.dW2, self.db2)
    
    def __mul__(self, other):
        return Grads(self.dW1 * other,
                     self.db1 * other,
                     self.dW2 * other,
                     self.db2 * other)
        
    

# Introducing interface of the Back Propagation class
class BackPropagationI(ABC):
    @staticmethod
    @abstractmethod
    def propagate(params: ParamsContainer, 
                  cache: ForwardPathContainer, 
                  X: np.ndarray, 
                  Y: np.ndarray
                ) -> Grads:
        ...

# Implementation of concrete class which will peform Back Propagation
class  BackPropagation(BackPropagationI):
    @staticmethod
    def propagate(params: ParamsContainer, 
                  cache: ForwardPathContainer, 
                  X: np.ndarray, 
                  Y: np.ndarray
                  ) -> Grads:
        
            m = X.shape[1]
            W1, W2 = params.W1, params.W2
            A1, A2 = cache.A1, cache.A2
            
            dZ2 = A2 - Y
            dW2 =( dZ2 @ A1.T) / m
            db2 = np.sum(dZ2, axis=1, keepdims=True) / m
            dZ1 = W2.T @ dZ2 * (1 - np.power(A1, 2))
            dW1 =( dZ1 @ X.T) / m
            db1 = (np.sum(dZ1, axis=1, keepdims=True)) / m
            
            logger.info('Gradients has been calculated succesfully')
            
            return Grads(dW1=dW1,
                         db1=db1,
                         dW2=dW2,
                         db2=db2)