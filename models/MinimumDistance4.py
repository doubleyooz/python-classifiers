import numpy as np

from models.Model import ModelInterface


class MinimumDistance4(ModelInterface):
    def __init__(self, class1_avg: list[float], class2_avg: list[float], classes=['setosa', 'versicolor'], columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'], point_names=['x1', 'x2', 'x3', 'x4']):
        self.class1_avg = class1_avg
        self.class2_avg = class2_avg
        self.classes = classes
        self.class_mapping = {class_label: idx for idx,
                              class_label in enumerate(self.classes)}
        self.point_names = point_names
        self.columns = columns
        self.metrics = {
            'fscore': 0,
            'kappa': 0,
            'matthews': 0,
            'precision': 0,
            'accuracy': 0,
            'recall': 0
        }
    '''
    def decision_function2(self, row: list[float]):
      x1 = row[0]
      x2 = row[1]
      x3 = row[2]
      x4 = row[3]
    
      class1_w0 = sum(val ** 2 for val in self.class1_avg)
      class2_w0 = sum(val ** 2 for val in self.class2_avg)

      d1 = x1 * self.class1_avg[0] + x2 * self.class1_avg[1] + x3 * self.class1_avg[2] + x4 * self.class1_avg[3] - (class1_w0 / 2)
      d2 = x1 * self.class2_avg[0] + x2 * self.class2_avg[1] + x3 * self.class2_avg[2] + x4 * self.class2_avg[3] - (class2_w0 / 2)

    
      return (d1 - d2)
    '''

    def decision_function(self, row: list[float]):
        x = np.array(row)

        class1_avg = np.array(self.class1_avg)
        class2_avg = np.array(self.class2_avg)

        class1_w0 = np.sum(class1_avg**2)
        class2_w0 = np.sum(class2_avg**2)

        d1 = np.dot(x, class1_avg) - (class1_w0 / 2)
        d2 = np.dot(x, class2_avg) - (class2_w0 / 2)

        return d1 - d2

    def classify(self, row):
        # print(self.classes[0] if self.decision_function(row) > 0 else self.classes[1])
        return self.classes[0] if self.decision_function(row) > 0 else self.classes[1]

    def surface(self, row):
        x1 = row[self.columns[0]]
        x2 = row[self.columns[1]]
        x3 = row[self.columns[2]]
        x4 = row[self.columns[3]]

        class1_w0 = sum(val ** 2 for val in self.class1_avg)
        class2_w0 = sum(val ** 2 for val in self.class2_avg)

        d1 = x1 * self.class1_avg[0] + x2 * self.class1_avg[1] + x3 * \
            self.class1_avg[2] + x4 * self.class1_avg[3] - (class1_w0 / 2)
        d2 = x1 * self.class2_avg[0] + x2 * self.class2_avg[1] + x3 * \
            self.class2_avg[2] + x4 * self.class2_avg[3] - (class2_w0 / 2)

        return (d1 + d2) / 2

    def predict(self, point):
        if point is None:
            raise ValueError("point cannot be None")
        row = [point.get(point_name) for point_name in self.point_names]
        return self.classify(row)

    def get_equation(self):
        return f'Decision Boundary Equation: x1 * {round(self.class1_avg[0], 2)} + x2 * {round(self.class1_avg[1], 2)} + x3 * {round(self.class1_avg[2], 2)} + x4 * {round(self.class1_avg[3], 2)} - {round(((self.class2_avg[0] * self.class2_avg[0]) + (self.class2_avg[1] * self.class2_avg[1]) + (self.class2_avg[2] * self.class2_avg[2]) + (self.class2_avg[3] * self.class2_avg[3])) / 4, 2)}'
