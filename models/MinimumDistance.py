class MinimumDistance:
    def __init__(self, setosa_avg, versicolor_avg):
        self.setosa_avg = setosa_avg
        self.versicolor_avg = versicolor_avg

    def decision_function(self, row):
        x1 = row['Sepal length']
        x2 = row['Sepal width']
        x3 = row['Petal length']
        x4 = row['Petal width']

        setosa_w0 = sum(val ** 2 for val in self.setosa_avg)
        versicolor_w0 = sum(val ** 2 for val in self.versicolor_avg)

        d1 = x1 * self.setosa_avg[0] + x2 * self.setosa_avg[1] + x3 * self.setosa_avg[2] + x4 * self.setosa_avg[3] - (setosa_w0 / 2)
        d2 = x1 * self.versicolor_avg[0] + x2 * self.versicolor_avg[1] + x3 * self.versicolor_avg[2] + x4 * self.versicolor_avg[3] - (versicolor_w0 / 2)


        return d1 - d2

    def classify(self, row):
        return 'setosa' if self.decision_function(row) > 0 else 'versicolor'

    def surface(self, row):
        x1 = row['Sepal length']
        x2 = row['Sepal width']
        x3 = row['Petal length']
        x4 = row['Petal width']

        setosa_w0 = sum(val ** 2 for val in self.setosa_avg)
        versicolor_w0 = sum(val ** 2 for val in self.versicolor_avg)

        d1 = x1 * self.setosa_avg[0] + x2 * self.setosa_avg[1] + x3 * self.setosa_avg[2] + x4 * self.setosa_avg[3] - (setosa_w0 / 2)
        d2 = x1 * self.versicolor_avg[0] + x2 * self.versicolor_avg[1] + x3 * self.versicolor_avg[2] + x4 * self.versicolor_avg[3] - (versicolor_w0 / 2)

        return (d1 + d2) / 2

    def predict(self, point):
        row = {'Sepal length': point['x1'], 'Sepal width':  point['x2'], 'Petal length':  point['x3'], 'Petal width':  point['x4']}
        return self.classify(row)

    def get_equation(self):
      return f'Decision Boundary Equation: x1 * {round(self.setosa_avg[0], 2)} + x2 * {round(self.setosa_avg[1], 2)} + x3 * {round(self.setosa_avg[2], 2)} + x4 * {round(self.setosa_avg[3], 2)} - {round(((self.versicolor_avg[0] * self.versicolor_avg[0]) + (self.versicolor_avg[1] * self.versicolor_avg[1]) + (self.versicolor_avg[2] * self.versicolor_avg[2]) + (self.versicolor_avg[3] * self.versicolor_avg[3]))/ 4, 2)}'

    def get_grid_values(self, data_df, columns):

      x1_values = np.linspace(data_df[columns[0]].min(), data_df[columns[0]].max(), 100)
      x2_values = np.linspace(data_df[columns[1]].min(), data_df[columns[1]].max(), 100)
      x3_values = np.linspace(data_df[columns[2]].min(), data_df[columns[2]].max(), 100)
      x4_values = np.linspace(data_df[columns[3]].min(), data_df[columns[3]].max(), 100)

      x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
      x3_grid, x4_grid = np.meshgrid(x3_values, x4_values)
      return {'x1': x1_grid, 'x2': x2_grid, 'x3': x3_grid, 'x4':  x4_grid}

    def get_decision_values(self, grid, columns):

      values = np.array([self.decision_function({columns[0]: x1, columns[1]: x2, columns[2]: x3, columns[3]: x4}) for x1, x2, x3, x4 in zip(np.ravel(grid['x1']), np.ravel(grid['x2']), np.ravel(grid['x3']), np.ravel(grid['x4']))])
      return values.reshape(grid['x1'].shape)
