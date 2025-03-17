import numpy as np
from logger import logger
from utils import type_check


def random_params_initialization(input_layer_size: int, 
                                 hidden_layers_size: int, 
                                 output_layers_size: int
                                 ) -> dict:
    '''
    Function which will initialize parameters base on provided sizes
    
    Args:
		input_layer_size (int): size of input layer 
		hidden_layers_size (int): size of hidder layer
		output_layers_size (int): size of output layer
  
	Raises:
		TypeError: if provided sizes is not of type int
  
	Returns:
		dict: dictionary wich initialized parameters of provided shapes
    '''
    
    for sample in (input_layer_size, hidden_layers_size,output_layers_size):
        type_check(sample, int)
        
    parameters = {"W1": np.random.randn(input_layer_size, hidden_layers_size).T,
                  "b1": np.zeros((hidden_layers_size, 1)),
                  "W2": np.random.randn(output_layers_size, hidden_layers_size),
                  "b2":  np.zeros((output_layers_size, 1))}
    
    logger.info('Parameters has been initialized')
    return parameters


random_params_initialization(10,10,10)
        
    