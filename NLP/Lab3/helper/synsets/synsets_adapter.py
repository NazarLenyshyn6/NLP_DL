from abc import ABC, abstractmethod
from typing import Any
import logging


# Interface of Synsets Adapter
class SynsetsAdapter(ABC):
    @abstractmethod
    def __init__(self, synsets_processor) -> None:
        ...
       
    @abstractmethod 
    def synsets(self, *args, **kwargs) -> Any:
        ...
        
        
# Implementation of Sysets Adapter for NLTK
class NLTKSynsetsAdapter(SynsetsAdapter):
    def __init__(self, synsets_processor) -> None:
        self.synsets_processor = synsets_processor
        
    def synsets(self, token: str, pos: str = 'v') -> Any:
        try:
            return self.synsets_processor.synsets(token, pos)
            
        except Exception as e:
            logging.exception(e)
            
    def __repr__(self) -> str:
        return f"NLTKSynsetsAdapter(synsets_processor={self.synsets_processor})"
    
    
if __name__ == '__main__':
    synsets_adapter = NLTKSynsetsAdapter('dummy_synsets_processor')
    print(synsets_adapter)