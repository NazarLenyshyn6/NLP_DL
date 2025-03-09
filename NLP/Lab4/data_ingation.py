from abc import ABC, abstractmethod
import pandas as pd
import logging

# Interface of Data Ingation
class DataIngationByPath(ABC):
  @abstractmethod
  def load(self, path:str) -> pd.DataFrame:
    ...


# Implementation of csv data ingation
class CSVDataIngation(DataIngationByPath):
  
  def __init__(self, samples_treshold: int = 10000) -> None:
    self.samples_treshold = samples_treshold

  def _large_dataset(self, path: str) -> None:
    return pd.read_csv(path, nrows=self.samples_treshold + 1).shape[0] > self.samples_treshold

  def load(self, path: str) -> pd.DataFrame | pd.io.parsers.readers.TextFileReader:
    try:
      if self._large_dataset(path):
          data = pd.read_csv(path,  chunksize=self.samples_treshold)
          logging.info('Data chuncks has been loaded successfully.')
          return data
      
      logging.info('Data has been loaded successfully.')
      return pd.read_csv(path)

    except Exception as e:
      logging.exception(e)

  def __repr__(self) -> str:
    return f"CSVDataIngation(samples_treshold={self.samples_treshold})"


if __name__ == '__main__':
    loader = CSVDataIngation()
    print(loader.load('/Users/nazarlenisin/Desktop/NLP_CV_UNIVERSITY/NLP/Lab4/data/data.csv'))
    
    loader =  CSVDataIngation(samples_treshold=1000)
    print(list(loader.load('/Users/nazarlenisin/Desktop/NLP_CV_UNIVERSITY/NLP/Lab4/data/data.csv')))
    
    print(list(loader.load('hello.csv')))
    
    
    
    