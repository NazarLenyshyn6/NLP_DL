from logger import logger
from typing import Any


def type_check(sample: Any, valid_type: Any) -> bool:
    ''' 
    function which validate if input sample of required type
    
    Args:
		sample (Any): instance which type has to be validated
		valid_type (Any): valid type
  
	Returns:
		bool: returns if sample has valid type or not
    '''
    
    if not isinstance(sample, valid_type):
        logger.info('Sample  has invalid type of %s | valid type %s', type(sample), valid_type)
        raise TypeError('Invalid type of imput sample | valid types: np.ndarray')
      
    logger.info('Sample has valid type')
