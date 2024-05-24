import numpy as np
from models.Model import ModelInterface
from utils import _unit_step_func

class NaivesBayes(ModelInterface):
    def __init__(self, learning_rate=0.01, max_iters=1000, pairs=['setosa', 'versicolor'], columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'], point_names=['x1', 'x2', 'x3', 'x4']):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.pairs = pairs
        self.bias = None
        self.columns = columns
        self.point_names = point_names
        self.errors = []
        self.weights = None


    def classify(self, row):
        result = [row[col] for col in self.columns]

        return self.pairs[0] if self.decision_function(result) > 0 else self.pairs[1]


    def get_equation(self):
        if self.weights is None:
            raise ValueError("Perceptron weights have not been trained yet.")

        # Extract weights and bias
        weights_str = " + ".join([f"{round(w, 2)} * x{i+1}" for i, w in enumerate(self.weights)])
        bias_str = f"- {round(self.weights[-1], 2)}" if self.weights[-1] != 0 else ""

        return f"Decision Boundary Equation: {weights_str} {bias_str}"


    def predict(self, point):
        row = {col: point[point_name] for col, point_name in zip(self.columns, self.point_names)}
        return self.classify(row)


    def predict_point(self, x, y):
        linear_output = np.dot([x,y], self.weights)
        y_predicted = self.decision_function(linear_output)
        return y_predicted


    def get_decision_values(self, grid):
        result = [self.decision_function([x1, x2, x3, x4]) for x1, x2, x3, x4 in zip(*[np.ravel(grid[name]) for name in self.point_names])]

        array_2d = np.array(result).reshape((100, 100))
        return array_2d


    def decision_function(self, x):
        linear_output = np.dot(x, self.weights) + self.bias# x * w0, missing bias

        return _unit_step_func(linear_output)


    def get_grid_values(self, data_df, columns):

        x1_values = np.linspace(data_df[columns[0]].min(), data_df[columns[0]].max(), 100)
        x2_values = np.linspace(data_df[columns[1]].min(), data_df[columns[1]].max(), 100)
        x3_values = np.linspace(data_df[columns[2]].min(), data_df[columns[2]].max(), 100)
        x4_values = np.linspace(data_df[columns[3]].min(), data_df[columns[3]].max(), 100)

        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
        x3_grid, x4_grid = np.meshgrid(x3_values, x4_values)
        return {'x1': x1_grid, 'x2': x2_grid, 'x3': x3_grid, 'x4':  x4_grid}
    
    # calculate all prior probabilities. We're dividing the number of samples where Y = y by the total of samples
    def calculate_prior(df, Y):
        classes = sorted(list(df[Y].unique()))
        prior = []
        for i in classes:
        prior.append(len(df[df[Y] ==i])/len(df))
        return prior

    def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
        feat = list(df.columns) # it extracts all features names then
        df = df[df[Y] == label] # it extracts all the datapoints where the Y value is the given label
        mean, std = df[feat_name].mean(), df[feat_name].std() # mean and standard deviation
        p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std) * np.exp(-((feat_val-mean)**2) / (2 * std**2))) # normal distribution
        return p_x_given_y

    def naive_bayes_gaussian(df, X, Y):
        # get feature names
        features = list(df.columns[: -1])

        #calculate the probability for each class
        prior = calculate_prior(df, Y)

        Y_pred = []

        #loop over every data sample
        for x in X:
            # calculate likelihood
            labels = sorted(list(df[Y].unique()))
            labels_length = len(labels)
            likelihood = [1]* labels_length

            for j in range(labels_length): # loop over every class

            for i in range(len(features)):   # loop over every feature
                # multiply individual conditional probabilities to get the likelihood
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

            # calculate posterior probability (numerator only)
            post_prob = [1]*labels_length

            # loop over all classes
            for j in range(labels_length):
            post_prob[j] = likelihood[j] * prior[j]

            Y_pred.append(np.argmax(post_prob)) # add to the returning array the class where the posterior probability is the highest

        return np.array(Y_pred)