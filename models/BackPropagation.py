import numpy as np
import pandas as pd

from models.Model import ModelInterface

class BackPropagation(ModelInterface):
    def __init__(self, df, class_column='Species', hidden_layers=[3, 3], columns_ignored=-1):
        df_copy = df.copy()        
        self.columns = list(df_copy.columns[: columns_ignored])
        self.point_names = ['x' + str(i) for i in range(1, len(self.columns) + 1)]
        self.classes = sorted(list(df_copy[class_column].unique()))
        self.class_mapping = {class_label: idx for idx, class_label in enumerate(self.classes)}
        #parameters      
        self.input_layer = len(self.columns)
        self.output_layer = len(self.classes)
        self.hidden_layers = hidden_layers
        
        layers = [ self.input_layer ] + self.hidden_layers + [self.output_layer]
        
        #weights
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
      
    # forward_propagate
    def decision_function(self, inputs):
        """Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropagation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropagation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

      
        return error

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):       
        return 1.0/(1.0 + np.exp(-x))
    
    def classify(self, row):
        # print(self.classes[0] if self.decision_function(row) > 0 else self.classes[1])
       
        if isinstance(row, pd.Series):
            row = row.tolist()
        elif isinstance(row, dict):
          
            row = list(row.values())
        
        return list(self.class_mapping.keys())[list(self.class_mapping.values()).index(round(self.decision_function(row)[0]))]
      
    
    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            
            derivatives = self.derivatives[i]         

            weights += derivatives * learning_rate

    # train
    def fit(self, inputs, targets, epochs, learning_rate, verbose=False):
        print(f'Learning rate: {learning_rate}, epochs: {epochs}')
        for i in range(epochs):
            sum_error = 0
            if i % 10 == 0:
                print(f'{i}/{epochs}')
            for j, (input, target) in enumerate(zip(inputs, targets)):
                
                # forward propagation
                output = self.decision_function(input)
             
                # calculate error
                error = target - output 
              
                # back propagation
                self.back_propagate(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)
            if verbose:
                print(f'Weights: {self.weights}')
                print(f'Error: {sum_error / len(inputs)} at epoch {i}')
            elif i == epochs - 1:
                print(f'Weights: {self.weights}')
                print(f'Error: {sum_error / len(inputs)} at epoch {i}')

    def _mse(self, target, output):
        return np.average((target - output)**2)
    
    def get_equation(self):
       
        str = 'yet to add the equation'

        return f"Decision Boundary Equation: {str}"
    
    def predict(self, point):
        return self.classify(point)


    def get_decision_values(self, grid):
        result = [(np.argmax(self.decision_function([x1, x2, x3, x4]))) for x1, x2, x3, x4 in zip(*[np.ravel(grid[name]) for name in self.point_names])]
        values = np.array(result)
        array_2d = values.reshape((100, 100))
        return array_2d
