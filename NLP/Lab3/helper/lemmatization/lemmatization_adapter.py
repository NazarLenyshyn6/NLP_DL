from abc import ABC, abstractmethod
from typing import Any
import logging


# Interface of Lemmatization Adapter
class LemmatizationAdatper(ABC):
  @abstractmethod
  def __init__(self, lemmatizer) -> None:
    ...

  @abstractmethod
  def lemmatize(self, *args, **kwargs) -> Any:
    ...


# Implementation of Lemmatization Adapter for NLTK library
class NLTKLemmatizationAdatper(LemmatizationAdatper):
  def __init__(self, lemmatizer):
    self.lemmatizer = lemmatizer

  def lemmatize(self, *args, **kwargs):
    try:
      return self.lemmatizer.lemmatize(*args, **kwargs)

    except Exception as e:
      logging.exception(e)

  def __repr__(self):
    return f"NLTKLemmatizationAdatper(lemmatizer={self.lemmatizer})"
  
  
if __name__ == '__main__':
  lemmatization_adapter = NLTKLemmatizationAdatper('dummy_lemmatizer')
  print(lemmatization_adapter)