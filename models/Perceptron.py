import numpy as np
from models.Model import ModelInterface

class Perceptron(ModelInterface):
    def __init__(self, learning_rate=0.01, n_iters=1000, pairs=['setosa', 'versicolor']):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.pairs = pairs
        self.errors = []
        self.weights = None

    def fit2(self,dataframe):
        data = dataframe.copy()
        x = data.iloc[0:100, [0, 1, 2, 3]].values
        data = data.iloc[0:100, 4].values
        data = np.where(data == self.pairs[0], -1, 1)
        self.fit(x, data)

    def fit(self, x, y):
        n_samples, n_features = x.shape

        # init parameters
        self.weights = np.zeros(n_features)


        y_ = np.array([1 if i > 0 else -1 for i in y])

        for _ in range(self.n_iters):
            error = 0
            for index, x_i in enumerate(x):
                # Ensure x_i is a numpy array
                if type(x_i) is not np.ndarray:
                  x_i = np.array(x_i)

                linear_output = np.dot(x_i, self.weights) # x * w0
                y_predicted = self.decision_function(linear_output)

                # Perceptron update rule
                if y_predicted <= 0:
                  update = self.weights if y_[index] == 0 else self.weights + self.lr * x_i * 1

                else:
                  update = self.weights if y_[index] == 1 else self.weights + self.lr * x_i * -1
                self.weights = update
                print(update)
                error += int(update != 0)
            self.errors.append(error)
        return self
   
    def get_equation(self):
        if self.weights is None:
            raise ValueError("Perceptron weights have not been trained yet.")

        # Extract weights and bias
        weights_str = " + ".join([f"{round(w, 2)} * x{i+1}" for i, w in enumerate(self.weights)])
        bias_str = f"- {round(self.weights[-1], 2)}" if self.weights[-1] != 0 else ""

        return f"Decision Boundary Equation: {weights_str} {bias_str}"

    def predict_point(self, x, y):
        linear_output = np.dot([x,y], self.weights)
        y_predicted = self.decision_function(linear_output)
        return y_predicted
    
    def classify(self, row):
      return self.pairs[0] if self.decision_function(row) > 0 else self.pairs[1]

    def predict(self, point):
      row = {'Sepal length': point['x1'], 'Sepal width':  point['x2'], 'Petal length':  point['x3'], 'Petal width':  point['x4']}
      return self.classify(row)
    
    def get_decision_values(self, grid, columns):
      values = np.array([self.decision_function({columns[0]: x1, columns[1]: x2, columns[2]: x3, columns[3]: x4}) for x1, x2, x3, x4 in zip(np.ravel(grid['x1']), np.ravel(grid['x2']), np.ravel(grid['x3']), np.ravel(grid['x4']))])
      array_2d = values.reshape((100, 100))
      return array_2d
    

    def decision_function(self, x):
        linear_output = np.dot(x, self.weights)
        # y_predicted = self.decision_function(linear_output)
        return self._unit_step_func(linear_output)

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, -1)

    def get_grid_values(self, data_df, columns):

      x1_values = np.linspace(data_df[columns[0]].min(), data_df[columns[0]].max(), 100)
      x2_values = np.linspace(data_df[columns[1]].min(), data_df[columns[1]].max(), 100)
      x3_values = np.linspace(data_df[columns[2]].min(), data_df[columns[2]].max(), 100)
      x4_values = np.linspace(data_df[columns[3]].min(), data_df[columns[3]].max(), 100)

      x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
      x3_grid, x4_grid = np.meshgrid(x3_values, x4_values)
      return {'x1': x1_grid, 'x2': x2_grid, 'x3': x3_grid, 'x4':  x4_grid}
