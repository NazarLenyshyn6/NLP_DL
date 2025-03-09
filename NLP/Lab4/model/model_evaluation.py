from typing import List
from abc import ABC, abstractmethod
from NLP.Lab4.model.model_builder import CustomModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Interfac of Model Evaluation class
class ModelEvaluationI(ABC):
    @abstractmethod
    def evaluate(self, model: CustomModel, X_test: List, y_test: List)  -> None:
        ...
        
# Implementation of concrete class which will evaluate model
class SimpleModelEvaluation(ModelEvaluationI):
    
    @staticmethod
    def _classification_report(y_true: List, y_pred: List) -> None:
        print('Classification report:')
        print(classification_report(y_true, y_pred))
        logging.info('Classification report done.')
        
    @staticmethod
    def _confusion_matrix(y_true: List, y_pred: List) -> None:
        plt.figure(figsize=(4,4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, vmin=0, cbar=False, fmt='g', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion matrix')
        plt.show()
        logging.info('Confusion matrix done')
        
    def evaluate(self, model: CustomModel, X_test: List, y_test: List) -> None:
        print('Model to Evaluate:', model)
        y_pred = model.predict(X_test)
        self._classification_report(y_test, y_pred)
        self._confusion_matrix(y_test, y_pred)
        
    def __repr__(self) -> str:
        return f"SimpleModelEvaluation()"
    
if __name__ == '__main__':
    model_evaluation = SimpleModelEvaluation()
    print(model_evaluation)