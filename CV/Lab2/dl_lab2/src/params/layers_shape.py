import numpy as np
from logger import logger
from abc import ABC, abstractmethod
from utils import type_check


# Introcuding Interfac of the object which retrive layes size
class LayersSizeRetrival(ABC):
    @staticmethod
    @abstractmethod
    def shape(X: np.ndarray, Y: np.ndarray, hidden_layer_size: int = 4):
        ...
        
# Implementation of concrete class which will retrive hidden layers size
class SimpleLayersSizeRetrival(LayersSizeRetrival):
    @staticmethod
    def shape(X: np.ndarray, Y: np.ndarray, hidden_layer_size: int = 4):
        for sample, valid_type in ((X, np.ndarray),(Y, np.ndarray),(hidden_layer_size,int)):
            type_check(sample, valid_type)
            
        return X.shape[0],hidden_layer_size, Y.shape[0]
    
if __name__ == '__main__':
    layers_retrival = SimpleLayersSizeRetrival()
    print(layers_retrival.shape(np.zeros((5,5)), np.zeros((5,5)), 10))
    