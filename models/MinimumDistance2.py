import numpy as np

from models.Model import ModelInterface


class MinimumDistance2(ModelInterface):
    def __init__(self, class1_avg, class2_avg, classes=['setosa', 'versicolor'], columns=['Sepal length', 'Sepal width'], point_names=['x1', 'x2']):
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

    def decision_function(self, row):
        x1 = row[0]
        x2 = row[1]

        # should have only two elements
        class1_w0 = sum(val ** 2 for val in self.class1_avg)
        class2_w0 = sum(val ** 2 for val in self.class2_avg)

        d1 = x1 * self.class1_avg[0] + x2 * \
            self.class1_avg[1] - (class1_w0 / 2)
        d2 = x1 * self.class2_avg[0] + x2 * \
            self.class2_avg[1] - (class2_w0 / 2)

        # print(d1, d2, d1-d2)
        return (d1 - d2)

    def classify(self, row):
        return self.classes[0] if self.decision_function(row) > 0 else self.classes[1]

    def surface(self, row):
        x1 = row[0]
        x2 = row[1]

        class1_w0 = sum(val ** 2 for val in self.class1_avg)
        class2_w0 = sum(val ** 2 for val in self.class2_avg)

        d1 = x1 * self.class1_avg[0] + x2 * \
            self.class1_avg[1] - (class1_w0 / 2)
        d2 = x1 * self.class2_avg[0] + x2 * \
            self.class2_avg[1] - (class2_w0 / 2)

        return (d1 + d2) / 2

    def predict(self, point):
        if point is None:
            raise ValueError("point cannot be None")
        row = [point.get(point_name) for point_name in self.point_names]
        return self.classify(row)

    def get_equation(self):
        return f'Decision Boundary Equation: x1 * {round(self.class1_avg[0], 2)} + x2 * {round(self.class1_avg[1], 2)} - {round(((self.class2_avg[0] * self.class2_avg[0]) + (self.class2_avg[1] * self.class2_avg[1])) / 4, 2)}'
