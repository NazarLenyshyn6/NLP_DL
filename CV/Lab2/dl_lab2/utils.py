import numpy as np
from logger import logger
from typing import Any

def type_check(sample: Any, valid_type: Any) -> TypeError | None:
    ''' Function which validate if input sample has required data type
    
    Args:
		sample (Any): sample which type will be validated.
		valid_type (Any): valid type of input sample.
  
	Raises:
		TypeError: occurs when input sample is not valid_type object
    '''
    
    if not isinstance(sample, valid_type):
        logger.info('Invalid type impute sample')
        raise TypeError('Input sample has to be of type %s instead got %s', valid_type, type(sample))
    
    logger.info('Valid type impute sample')
    
def sigmoid(X: np.ndarray) -> np.ndarray:
  type_check(X, np.ndarray)
  return 1 / (1 + np.e**(-X))
