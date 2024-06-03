from abc import ABC, abstractmethod

class ModelInterface(ABC):
    def __init__(self, class1_avg, class2_avg, pairs=['setosa', 'versicolor'], columns=['Sepal length', 'Sepal width']):
        self.class1_avg = class1_avg
        self.class2_avg = class2_avg
        self.pairs = pairs
        self.columns = columns

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

    @abstractmethod
    def get_decision_values(self, grid, columns):
        pass
