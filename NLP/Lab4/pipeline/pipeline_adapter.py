from abc import ABC, abstractmethod
import logging
from typing import Any, List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from NLP.Lab4.text_preprocessing.text_cleaning import TextCleaner, SimpleTextCleaner
from NLP.Lab4.text_preprocessing.text_tokenization import TextTokenizer, NLTKTextTokenier
from NLP.Lab4.text_preprocessing.text_embedding import TextEmbedding, Word2VecEmbedding


# Interface of Pipeline Adapterf
class PipelineAdapter(ABC, BaseEstimator, TransformerMixin):
    @abstractmethod
    def __init__(self, func) -> None:
        ...
        
    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        ...
        
    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        ...
        
        
# Implementatino of Piplene Adapter for text cleaning
class TextCleaningPipelineAdapter(PipelineAdapter):
    def __init__(self, func: TextCleaner) -> None:
        self.func = func
        
    def fit(self, texts: List, y=None) -> 'TextCleaningPipelineAdapter':
        return self
    
    def transform(self, text: List) -> List:
        return self.func.clean(text)
    
    def __repr__(self) -> str:
        return f"TextCleaningPipelineAdapter(cleaner={self.func})"
    
    
# Implementation of Pipeline Adapter for text tokenization
class TextTokenizerPipelineAdapter(PipelineAdapter):
    def __init__(self, func: TextTokenizer) -> None:
        self.func = func
    
    def fit(self, texts: List, y=None) -> 'TextTokenizerPipelineAdapter':
        return self
    
    def transform(self, texts: List) -> List:
        return self.func.tokenize(texts)
    
    def __repr__(self) -> str:
        return f"TextTokenizerPipelineAdapter(tokenizer={self.func})"
    
    
# Implementation of Pipeline Adapter for text embedding
class TextEmbeddingPipelineAdapter(PipelineAdapter):
    def __init__(self, func: TextEmbedding) -> None:
        self.func = func
        
    def fit(self, sequences: List[List[str]], y=None) -> None:
        self.func.fit(sequences)
        return self
        
    def transform(self, sequences: List[List[str]]) -> np.ndarray:
        return self.func.transform(sequences)
        
    def __repr__(self) -> str:
        return f"TextEmbeddingPipelineAdapter(embedding_model={self.func})"
        
    
if __name__ == '__main__':
    adapted_text_cleaning = TextCleaningPipelineAdapter(SimpleTextCleaner())
    print(adapted_text_cleaning)
    
    adapted_text_tokenizzer = TextTokenizerPipelineAdapter(NLTKTextTokenier())
    print(adapted_text_tokenizzer)
    
    adapted_text_embedding = TextEmbeddingPipelineAdapter(Word2VecEmbedding())
    print(adapted_text_embedding)
    