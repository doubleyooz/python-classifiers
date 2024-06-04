import numpy as np
import pandas as pd
import numpy as np
import random

path = './assets/Iris.csv'
df = pd.read_csv(path)

class BackPropagation(object):
    def __init__(self, df, class_column='Species', hidden_layers=[3, 3], columns_ignored=-1):
        df_copy = df.copy()        
        self.columns = list(df_copy.columns[: columns_ignored])
        self.point_names = ['x' + str(i) for i in range(1, len(self.columns) + 1)]
        self.classes = sorted(list(df_copy[class_column].unique()))
    
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
      
        
    def forward_propagate(self, inputs):
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

            # save the activations for backpropogation
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
    

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            
            derivatives = self.derivatives[i]         

            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate, verbose=False):
        for i in range(epochs):
            sum_error = 0
            for j, (input, target) in enumerate(zip(inputs, targets)):
                
                # forward propagation
                output = self.forward_propagate(input)
             
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
                print(f'Error: {sum_error / len(inputs)} at epoch {i}')

    def _mse(self, target, output):
        return np.average((target - output)**2)


def get_pairs(exclude = 'virginica', column='Species', pairs = ['virginica', 'setosa', 'versicolor'], generate_numbers = False, seed=None, overwrite_classes=False):
    
    print(f'exclude={exclude}')

    print(exclude, pairs)
    data_df = df[df[column] != exclude]
    opposite_data_df = df[df[column] == exclude]

    class1_df = df[df[column] == pairs[0]]
    class2_df = df[df[column] != pairs[0]]
        
            

    if seed is None:
        seed = random.randint(1, 1000)
    
    
    # Randomly sample 70% of your dataframe
    data = data_df.copy().sample(frac=0.7, random_state=seed)

    # Get the remaining 30% of the data
    test = data_df.copy().drop(data.index)

   
    if overwrite_classes:        
        data.loc[data[column] != pairs[0], column] = pairs[1]
        test.loc[test[column] != pairs[0], column] = pairs[1]
        class1_df.loc[class1_df[column] != pairs[0], column] = pairs[1]
        class2_df.loc[class2_df[column] != pairs[0], column] = pairs[1]
        opposite_data_df.loc[opposite_data_df[column] != pairs[0], column] = pairs[1]
    
        if data[column].nunique(0) == 1:
        
            if len(opposite_data_df) >= len(data):
                opposite_data_df_sliced = opposite_data_df.head(len(data))
                data = pd.concat([opposite_data_df_sliced, data])
                opposite_data_df_sliced = opposite_data_df.head(len(test))
                test = pd.concat([opposite_data_df_sliced, test])
            else: 
                data_sliced = data.head(len(opposite_data_df))
                data = pd.concat([data_sliced, opposite_data_df])
                test_sliced = test.head(len(opposite_data_df))
                test = pd.concat([test_sliced, opposite_data_df])
                
  

    print(f'Training data size: {len(data)}')
    print(f'Testing data size: {len(test)}')


    print(f'non_{exclude}_df size: {len(data_df)}')
    return data, test, class1_df, class2_df, opposite_data_df




if __name__ == "__main__":

    data, test, class1_df, class2_df, opposite_data_df = get_pairs(exclude='virginica')
    mlp = BackPropagation(df=data)
    print(f'input_layer = {mlp.input_layer}, hidden_layers = {mlp.hidden_layers}, output_layer = {mlp.output_layer}')
    print(f'weights {len(mlp.weights)} = {mlp.weights}')
    class_mapping = {mlp.classes[0]: 0, mlp.classes[1]: 1}
   
    Y_train = data.iloc[:, -1].values 
    Y_train = [class_mapping[class_name] for class_name in Y_train]
    X_train = data.iloc[:, :-1].values
    
    
    Y_test = test.iloc[:, -1].values 
    Y_test = [class_mapping[class_name] for class_name in Y_test]
    X_test = test.iloc[:, :-1].values
    

    # mlp.train(X_train, Y_train, 5000, 0.01, verbose=True)
    # for j, (x, y, output) in enumerate(zip(X_test, Y_test, outputs)):
        # print("Input: {} - Target: {} - Prediction: {}".format(x, y, output))

