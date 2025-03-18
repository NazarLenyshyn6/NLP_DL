import numpy as np
from logger import logger
from abc import ABC, abstractmethod
from typing import Any, Callable
from src.params.weights import WeightInitializerI, RandomWeightInitializer
from src.propagation.forward_propagation import ForwardPropagationI, ForwardPropagation
from src.propagation.back_propaagation import BackPropagationI, BackPropagation


# Intorducing model interfac
class Model:
    @abstractmethod
    def __init__(self,
                 params_initializer: WeightInitializerI = RandomWeightInitializer,
                 forward_propagation: ForwardPropagationI = ForwardPropagation,
                 back_propagation: BackPropagationI = BackPropagation,
                 cost: Any = None,
                 ) -> None:
        ...
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        ...
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
        
        
# implementatino of concrete nn
class NNModel(Model):
    def __init__(self,
                 params_initializer: WeightInitializerI = RandomWeightInitializer,
                 forward_propagation: ForwardPropagationI = ForwardPropagation,
                 back_propagation: BackPropagationI = BackPropagation,
                 cost: Callable = None,
                 ) -> None:
        self.params_initializer = params_initializer
        self.forward_propagation = forward_propagation
        self.back_propagation = back_propagation
        self.cost = cost
        
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray, 
            num_iters: int=100, 
            learning_rate: int =  0.001
            ):
        self.params = self.params_initializer.weights(input_size=X_train.shape[0], 
                                                hidden_size=4, 
                                                output_size=y_train.shape[0])
        
        for _ in range(num_iters):
            A2, forward_prop = self.forward_propagation.forward(X_train, self.params)
            eps = 1e-10 
            A2 = np.clip(A2, eps, 1 - eps)
            cost = -np.mean(y_train * np.log(A2) + (1 - y_train) * np.log(1 - A2))
            print(cost)
            back_prop  = self.back_propagation.propagate(self.params, forward_prop, X_train, y_train)
            self.params = self.params - (back_prop * learning_rate).values
            
        logger.info('Model trained successfully')
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.forward_propagation.forward(X, self.params)[0] > 0.5, 1, 0)
    
    def __repr__(self) -> str:
        return f'NNModel()'
    
if __name__ == '__main__':
    print(NNModel())