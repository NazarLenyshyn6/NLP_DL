from abc import ABC, abstractmethod
from NLP.Lab4.model.model_builder import SentimentModel
from typing import List
import logging
import optuna
import numpy as np


# Interface of Optuna Hyper Parameter Tunner
class OptunaTunner(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        ...
        
    @abstractmethod
    def objective(trial):
        ...
        
    def optimize(self,n_trials: int = 50, n_jobs=4, direction: str = 'maximize') -> None:
        self.study = optuna.create_study(direction=direction)
        logging.info('Study has been created')
        
        self.study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        logging.info('Parameters has been selected.')
        
    @property
    def best_params(self) -> dict:
        if not hasattr(self, 'study'):
            raise ValueError('Call .optimize() first.')
        
        return self.study.best_params
    
    @property
    def best_score(self) -> float:
        if not hasattr(self, 'study'):
            raise ValueError('Call .optimize() first.')
        
        return self.study.best_value
    
    
# Implementatino of Optuna Hyper Parameter Tunner for SentimentModel
class SentimentModelOptuneTunner(OptunaTunner):
    def __init__(self, 
                 X_train: List, 
                 y_train: List, 
                 X_test: List, 
                 y_test: List, 
                 params: dict | None = None,
                 funcs: dict | None = None
                 ) -> None:
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        if not params:
            raise ValueError('Input params dictionary can not be empty')
        self.params = params
        self.funcs = funcs
        
    def objective(self, trial):
        params_dict = {}
        
        for param_name, params in self.params.items():
            sub_params_dict = {}
            
            for sub_param_name, sub_param in params.items():
                sub_param_type, sub_param_value = sub_param
                
                if sub_param_type == 'int':
                    sub_params_dict[sub_param_name] = trial.suggest_int(sub_param_name,*sub_param_value)
                    
                elif sub_param_type == 'float':
                    sub_params_dict[sub_param_name] = trial.suggest_float(sub_param_name,*sub_param_value)
                    
                elif sub_param_type == 'categorical':
                    sub_params_dict[sub_param_name] = trial.suggest_categorical(sub_param_name, sub_param_value)
                    
                else:
                    raise ValueError('Param type has to be of type: int, float, categorical')
            
            params_dict[param_name] = sub_params_dict
            logging.info('Sub params dict has been formed.')
            
        logging.info('Params dict has been formed.')
        
        if self.funcs:
            params_dict.update(self.funcs)
            
        model = SentimentModel(**params_dict).fit(self.X_train, self.y_train)
        logging.info('Model has been initialized and fit')
        
        return np.mean(self.y_test == model.predict(self.X_test))
        
    def __repr__(self) -> str:
        return f"SentimentModelOptuneTunner(params={self.params})"
        
        
if __name__ == '__main__':
    model_hyperparams = {'text_embedding_params': {'dim': ('int', (10, 200)), 
                                                   'window':  ('int', (2,3))},
                         'estimator_params': {'penalty', ('categorical', ("l1", "l2"))}}
    
    optimizer = SentimentModelOptuneTunner(0,0,0,0, model_hyperparams)
    print(optimizer)
    print(optimizer.objective(1))
    

    