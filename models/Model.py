from abc import ABC, abstractmethod


class ModelInterface(ABC):

    @abstractmethod
    def decision_function(self, row):
        pass

    @abstractmethod
    def classify(self, row):
        pass

    @abstractmethod
    def predict(self, point):
        pass

    @abstractmethod
    def get_equation(self):
        pass
