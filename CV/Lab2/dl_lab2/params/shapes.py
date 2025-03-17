import numpy as np
from typing import Tuple
from logger import logger
from utils import type_check


def get_shapes(sample: np.ndarray) -> Tuple[int, int]:
    '''
    function which will return shape of input sample
    
    Args:
		samle (np.ndarray): sample which shapes has to be detected
  
	Raises:
		TypeError: when sample is not of type np.ndrray
  
	Returns:
		Tuple[int, int]: tuple of integers, which represent shape of input sample
    '''

    type_check(sample, np.ndarray)
    return sample.shape
  
  
def get_layer_sizes(X: np.ndarray, Y: np.ndarray, hidden_size: int = 4) -> None:
  '''
  Function which will return shape of input samples and hidden size
  
  Args:
    X (np.ndarray): Input features matrix
    Y (np.ndarray): Input target matrix
    hidde_size (int): number of neurons in hidden layer
    
  Raises:
    TypeError: if sample of invalid type
    
  Returns:
      Tuple[int,int,int]: dimensionality of ever single argument
  '''
  type_check(hidden_size, int)
  return get_shapes(X)[0], hidden_size, get_shapes(Y)[0]
  
  
get_layer_sizes(np.zeros((10,10)), np.zeros((10,10)), 15)

    