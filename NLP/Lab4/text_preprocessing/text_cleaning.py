from abc import ABC, abstractmethod
import logging
import re
from typing import List

# Interface of text cleaning
class TextCleaner(ABC):
    def clean(self, texts: List[str]) -> List[str]:
        ...
        
# Implementation of conctete text clearn
class SimpleTextCleaner(TextCleaner):
    def clean(self, texts: List[str]) -> str:
        if not isinstance(texts, list):
            raise TypeError('Intup texts has to be of type list')
        
        cleaned_text = [0] * len(texts)
        
        for text_idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError('Text has to be of type str')

            text = text.lower()
            text = re.sub('\d+', '', text)
            text = re.sub('[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            logging.info('Text has been cleaned sucsessfully.')
            cleaned_text[text_idx] = text
        return cleaned_text
    
    def __repr__(self) -> str:
        return f"SimpleTextCleaner()"
    
    
if __name__ == '__main__':
    text_cleaner = SimpleTextCleaner()
    print(text_cleaner)
    print(text_cleaner.clean(['HELlo woRld 124$ is  your price', 'What is you name', 'What is you name']))
    print(text_cleaner.clean(10))
    print(text_cleaner.clean([True]))
        