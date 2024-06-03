import numpy as np

from models.Model import ModelInterface

class MinimumDistance2(ModelInterface):
  def __init__(self, class1_avg, class2_avg, pairs=['setosa', 'versicolor'], columns=['Sepal length', 'Sepal width'], point_names=['x1', 'x2']):
    self.class1_avg = class1_avg
    self.class2_avg = class2_avg
    self.pairs = pairs    
    self.point_names = point_names
    self.columns = columns

  def decision_function(self, row):
      x1 = row[self.columns[0]]
      x2 = row[self.columns[1]]

      class1_w0 = sum(val ** 2 for val in self.class1_avg) # should have only two elements
      class2_w0 = sum(val ** 2 for val in self.class2_avg)

      d1 = x1 * self.class1_avg[0] + x2 * self.class1_avg[1] - (class1_w0 / 2)
      d2 = x1 * self.class2_avg[0] + x2 * self.class2_avg[1] - (class2_w0 / 2)

      # print(d1, d2, d1-d2)
      return (d1 - d2)

  def classify(self, row):
      return self.pairs[0] if self.decision_function(row) > 0 else self.pairs[1]

  def surface(self, row):
      x1 = row[self.columns[0]]
      x2 = row[self.columns[1]]       

      class1_w0 = sum(val ** 2 for val in self.class1_avg)
      class2_w0 = sum(val ** 2 for val in self.class2_avg)

      d1 = x1 * self.class1_avg[0] + x2 * self.class1_avg[1] - (class1_w0 / 2)
      d2 = x1 * self.class2_avg[0] + x2 * self.class2_avg[1] - (class2_w0 / 2)

      return (d1 + d2) / 2

  def predict(self, point):
    row = {col: point[point_name] for col, point_name in zip(self.columns, self.point_names)}
    return self.classify(row)


  def get_equation(self):
    return f'Decision Boundary Equation: x1 * {round(self.class1_avg[0], 2)} + x2 * {round(self.class1_avg[1], 2)} - {round(((self.class2_avg[0] * self.class2_avg[0]) + (self.class2_avg[1] * self.class2_avg[1]))/ 4, 2)}'


  def get_decision_values(self, grid):   
    print(grid)
    values = np.array([self.decision_function({self.columns[0]: x1, self.columns[1]: x2}) for x1, x2, in zip(np.ravel(grid['x1']), np.ravel(grid['x2']))])
    array_2d = values.reshape((100, 100))
    return array_2d
  
  def get_decision_values2(self, grid):
    values = np.array([self.surface({self.columns[0]: x1, self.columns[1]: x2}) for x1, x2, in zip(np.ravel(grid['x1']), np.ravel(grid['x2']))])
    array_2d = values.reshape((100, 100))
    return array_2d
  
