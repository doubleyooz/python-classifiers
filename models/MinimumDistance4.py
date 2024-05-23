import numpy as np

from models.Model import ModelInterface

class MinimumDistance4(ModelInterface):
    def __init__(self, class1_avg, class2_avg, pairs=['setosa', 'versicolor'], columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'], point_names=['x1', 'x2', 'x3', 'x4']):
      self.class1_avg = class1_avg
      self.class2_avg = class2_avg
      self.pairs = pairs    
      self.point_names = point_names
      self.columns = columns

    def decision_function(self, row):
      x1 = row[self.columns[0]]
      x2 = row[self.columns[1]]
      x3 = row[self.columns[2]]
      x4 = row[self.columns[3]]

      class1_w0 = sum(val ** 2 for val in self.class1_avg)
      class2_w0 = sum(val ** 2 for val in self.class2_avg)

      d1 = x1 * self.class1_avg[0] + x2 * self.class1_avg[1] + x3 * self.class1_avg[2] + x4 * self.class1_avg[3] - (class1_w0 / 2)
      d2 = x1 * self.class2_avg[0] + x2 * self.class2_avg[1] + x3 * self.class2_avg[2] + x4 * self.class2_avg[3] - (class2_w0 / 2)


      return (d1 - d2)

    def classify(self, row):
      return self.pairs[0] if self.decision_function(row) > 0 else self.pairs[1]

    def surface(self, row):
      x1 = row[self.columns[0]]
      x2 = row[self.columns[1]]
      x3 = row[self.columns[2]]
      x4 = row[self.columns[3]]


      class1_w0 = sum(val ** 2 for val in self.class1_avg)
      class2_w0 = sum(val ** 2 for val in self.class2_avg)

      d1 = x1 * self.class1_avg[0] + x2 * self.class1_avg[1] + x3 * self.class1_avg[2] + x4 * self.class1_avg[3] - (class1_w0 / 2)
      d2 = x1 * self.class2_avg[0] + x2 * self.class2_avg[1] + x3 * self.class2_avg[2] + x4 * self.class2_avg[3] - (class2_w0 / 2)

      return (d1 + d2) / 2

    def predict(self, point):
      row = {col: point[point_name] for col, point_name in zip(self.columns, self.point_names)}
      return self.classify(row)


    def get_equation(self):
      return f'Decision Boundary Equation: x1 * {round(self.class1_avg[0], 2)} + x2 * {round(self.class1_avg[1], 2)} + x3 * {round(self.class1_avg[2], 2)} + x4 * {round(self.class1_avg[3], 2)} - {round(((self.class2_avg[0] * self.class2_avg[0]) + (self.class2_avg[1] * self.class2_avg[1]) + (self.class2_avg[2] * self.class2_avg[2]) + (self.class2_avg[3] * self.class2_avg[3]))/ 4, 2)}'

    def get_grid_values(self, data_df, columns):

      x1_values = np.linspace(data_df[columns[0]].min(), data_df[columns[0]].max(), 100)
      x2_values = np.linspace(data_df[columns[1]].min(), data_df[columns[1]].max(), 100)
      x3_values = np.linspace(data_df[columns[2]].min(), data_df[columns[2]].max(), 100)
      x4_values = np.linspace(data_df[columns[3]].min(), data_df[columns[3]].max(), 100)

      x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
      x3_grid, x4_grid = np.meshgrid(x3_values, x4_values)
      return {'x1': x1_grid, 'x2': x2_grid, 'x3': x3_grid, 'x4':  x4_grid}

    def get_decision_values(self, grid):
      values = np.array([self.decision_function({self.columns[0]: x1, self.columns[1]: x2, self.columns[2]: x3, self.columns[3]: x4}) for x1, x2, x3, x4 in zip(np.ravel(grid['x1']), np.ravel(grid['x2']), np.ravel(grid['x3']), np.ravel(grid['x4']))])
      array_2d = values.reshape((100, 100))
      return array_2d
    
    def get_decision_values2(self, grid, columns):

      values = np.array([self.surface({columns[0]: x1, columns[1]: x2, columns[2]: x3, columns[3]: x4}) for x1, x2, x3, x4 in zip(np.ravel(grid['x1']), np.ravel(grid['x2']), np.ravel(grid['x3']), np.ravel(grid['x4']))])
      array_2d = values.reshape((100, 100))
      return array_2d
