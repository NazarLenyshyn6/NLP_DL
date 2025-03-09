from abc import ABC, abstractmethod
from typing import List, Any
from dataclasses import dataclass, field
from NLP.Lab4.text_preprocessing.text_cleaning import TextCleaner, SimpleTextCleaner
from NLP.Lab4.text_preprocessing.text_tokenization import TextTokenizer, NLTKTextTokenier
from NLP.Lab4.text_preprocessing.text_embedding import TextEmbedding, Word2VecEmbedding
from NLP.Lab4.pipeline.pipeline_adapter import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import logging



# Interface of the Modle
class CustomModel(ABC):
    def fit(self, *args, **kwargs) -> None:
        ...
        
    def predict(self, *args, **kwargs) -> None:
        ...
        
# Implementation of the model for Sentement classification

class SentimentModel(CustomModel):
    def __init__(self, 
                 text_cleaner: TextCleaner = SimpleTextCleaner,
                 text_tokenizer: TextTokenizer = NLTKTextTokenier,
                 text_embedding: TextEmbedding = Word2VecEmbedding,
                 estimator: Any = LogisticRegression,
                 text_cleaner_params: dict | None = None,
                 text_tokenizer_params: dict | None = None,
                 text_embedding_params: dict | None = None,
                 estimator_params: dict | None = None
                 ) -> None:
        
        text_cleaner = text_cleaner(**text_cleaner_params) if text_cleaner_params else text_cleaner()
        text_tokenizer = text_tokenizer(**text_tokenizer_params) if text_tokenizer_params else text_tokenizer()
        text_embedding = text_embedding(**text_embedding_params) if text_embedding_params else text_embedding()
        estimator = estimator(**estimator_params) if estimator_params else estimator()

        self._pipeline = Pipeline(steps=[
			('text_cleaning', TextCleaningPipelineAdapter(text_cleaner)),
   			('text_tokenization', TextTokenizerPipelineAdapter(text_tokenizer)),
			('text_embedding', TextEmbeddingPipelineAdapter(text_embedding)),
			('estimator', estimator)
		])
        
    def fit(self, X_train: List, y_train: List) -> None:
        self._pipeline.fit(X_train, y_train)
        return self
            
    def predict(self, X: List) -> List:
        return self._pipeline.predict(X)
        
    def __repr__(self) -> str:
        print('Sentiment model:')
        print('Pipeline:', self._pipeline)
        return ''
    
if __name__ == '__main__':
    model = SentimentModel(text_embedding_params={'dim': 12})
    print(model)