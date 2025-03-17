import numpy as np
from typing import Tuple
from logger import logger



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
    
    if not isinstance(sample, np.ndarray):
        raise TypeError('Invalid type of imput sample | valid types: np.ndarray')
    
    return sample.shape
    