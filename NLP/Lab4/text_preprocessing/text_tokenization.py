from abc import ABC, abstractmethod
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from typing import List
import logging

# Intoducing interface of text Tokenier
class TextTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: List[str]) -> List:
        ...
        
        
# Implementation on nltk tokenier
class NLTKTextTokenier(TextTokenizer):
    def __init__(self, tokenizer = word_tokenize) -> None:
        self.tokenizer = tokenizer
        
    def tokenize(self, texts: List[str]) -> List:
        if not isinstance(texts, list):
            raise TypeError('Input texts has to be of type list')
        
        tokenized_texts = [0] * len(texts)
        
        for text_idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError('Input text has to be of type str')
            
            try:
                tokenized_texts[text_idx] = self.tokenizer(text)
                
            except Exception as e:
                logging.exception(e)
                raise e
        logging.info('Texts has been tokenied.')
        return tokenized_texts
            
    def __repr__(self) -> str:
        return f"NLTKTextTokenier(tokenizer={self.tokenizer})"
            
if __name__ == '__main__':
    text = ['hello what is you name', 'How are YOu?']
    tokenizer = NLTKTextTokenier()
    print(tokenizer)
    print(tokenizer.tokenize(text))