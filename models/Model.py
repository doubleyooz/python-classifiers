from abc import ABC, abstractmethod

class ModelInterface(ABC):
    def __init__(self, setosa_avg, versicolor_avg):
        self.setosa_avg = setosa_avg
        self.versicolor_avg = versicolor_avg

    @abstractmethod
    def decision_function(self, row):
        pass


    @abstractmethod
    def classify(self, row):
        pass

    @abstractmethod
    def surface(self, row):
        pass

    @abstractmethod
    def predict(self, point):
        pass

    @abstractmethod
    def get_equation(self):
        pass

    @abstractmethod
    def get_grid_values(self, data_df, columns):
        pass

    @abstractmethod
    def get_decision_values(self, grid, columns):
        pass
