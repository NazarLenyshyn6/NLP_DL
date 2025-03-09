from abc import ABC, abstractmethod
import logging
from typing import List, Any
from NLP.Lab4.text_preprocessing.text_tokenization import NLTKTextTokenier
from gensim.models import Word2Vec, KeyedVectors
from gensim.models import KeyedVectors
import spacy
import numpy as np
logging.basicConfig(level=logging.INFO)


# Interface of Text Embedding
class TextEmbedding(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        ...
        
    @abstractmethod
    def fit(self, sequences: List[List[str]]) -> None:
        ...
        
    @abstractmethod
    def transform(self, sequences: List[List[str]]) -> np.ndarray:
        ...
        
        

# Implementation of Word2Vec Average Word Embedding
class Word2VecEmbedding(TextEmbedding):
    def __init__(self,  dim: int = 100, window: int = 2):
        self.dim = dim
        self.window = window
        
    def fit(self, sequences: List[List[str]]) -> None:
        
        try:
            self.model = model = Word2Vec(sequences, vector_size=self.dim, window=self.window, min_count=1, sg=1)
            logging.info('Word2Vec model has been trained.')
            
        except Exception as e:
            logging.exception(e)
            
    def transform(self, sequences: List[List[str]]) -> np.ndarray:
        if not hasattr(self, 'model'):
            raise ValueError("Model is not trained. Call `fit()` first.")
        
        if not sequences:
            raise ValueError('Can not transform empy list.')

        vectorized_texts = [0] * len(sequences)

        for sequence_idx, sequence in enumerate(sequences):
          vectorized_tokens = np.zeros(self.dim)

          for token in sequence:
            if token in self.model.wv:
              vectorized_tokens += self.model.wv[token]

          if len(sequence): vectorized_tokens / len(sequence) 
          vectorized_texts[sequence_idx] = vectorized_tokens

        logging.info('Texts has been embedded.')
        return vectorized_texts
        
    def __repr__(self) -> str:
        return f"Word2VecEmbedding(model={Word2Vec})"
    
    
# Impementatoin of Word2Vec using Google pretrained Word2Vec
class SpacyWord2Vec:
    def __init__(self):
        self.model = spacy.load('en_core_web_md')
        
    def fit(self, sequences: Any = None) -> 'SpacyWord2Vec':
        return self
    
    def transform(self, sequences: List[List[str]]) -> np.ndarray:
        if not sequences:
            raise ValueError('Can not transform empy list.')

        vectorized_texts = [0] * len(sequences)
        
        for sequence_idx, sequence in enumerate(sequences):
            vectorized_tokens = np.zeros(300)
            
            for token in sequence:
                if token in self.model.vocab:
                    vectorized_tokens += self.model.vocab[token].vector
            if len(sequence): vectorized_tokens / len(sequence)
            vectorized_texts[sequence_idx] = vectorized_tokens
            
        logging.info('Texts has been embedded.')
        return vectorized_texts

        
    def __repr__(self) -> str:
        return f"SpacyWord2Vec(model={self.model})"
    
        
if __name__ == '__main__':
    
    # # tokenize 
    text = ['hello my friend what is you name', 'what is you name']
    tokenizer = NLTKTextTokenier()
    tokenized_text = tokenizer.tokenize(text)
    
    # embedding
    embedding_model = Word2VecEmbedding()
    embedding_model.fit(tokenized_text)
    print(embedding_model.transform(tokenized_text))
    
    spacy_embedding = SpacyWord2Vec()
    print(spacy_embedding.model)
    print(spacy_embedding)
    print(spacy_embedding.fit(None))
    print(spacy_embedding.transform(tokenized_text))
    
    
