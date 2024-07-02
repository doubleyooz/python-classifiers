import numpy as np
from models.Model import ModelInterface
from utils.npHelper  import _unit_step_func

class Perceptron(ModelInterface):
  def __init__(self, df, class_column='Species', columns_ignored=-1):
   
    df_copy = df.copy() 
    self.bias = 0
    self.columns = list(df_copy.columns[: columns_ignored])
    self.classes = sorted(list(df_copy[class_column].unique()))
    self.class_mapping = {class_label: idx for idx, class_label in enumerate(self.classes)}
    self.point_names = ['x' + str(i) for i in range(1, len(self.columns) + 1)]
    self.errors = []
    self.weights = None
    self.metrics = {
        'fscore': 0,
        'kappa': 0,
        'matthews': 0,
        'precision': 0,
        'accuracy': 0,
        'recall': 0
      }
    self.weights = np.zeros(len(self.columns))
    self.bias = 0
       
  def classify(self, row):
    result = [row[col] for col in self.columns]

    return self.classes[0] if self.decision_function(result) > 0 else self.classes[1]

  def fit(self, inputs, targets, learning_rate=0.01, epochs=10000, plateau_tolerance = 20, reset_weights = False, verbose=False):
    max_lr = 0.9
    min_lr = 0.001
    if learning_rate > 1:
        learning_rate = max_lr
    elif learning_rate <= 0:
        learning_rate = min_lr

    if reset_weights:
      self.weights = np.zeros(len(self.columns))
      self.bias = 0

    # init parameters      
    current_interaction = 0
    plateau = 0
    disturbance = 0
        
                       
    while current_interaction < epochs and plateau < len(inputs) - 2:
      error = 0
     
      for index, x_i in enumerate(inputs):
        # Ensure x_i is a numpy array
        if type(x_i) is not np.ndarray:
          x_i = np.array(x_i)
  
        y_predicted = self.decision_function(x_i)
                #             1            1 =  0
                #             0            0 =  0
                #             1            0 =  1
                #             0            1 = -1
        update = learning_rate * (targets[index] - y_predicted) 
            
        self.weights += update * x_i + self.bias
        self.bias = update
       
        if y_predicted != targets[index]:
         
          plateau = 0
          disturbance += 1
          
        else:
          plateau += 1 
          disturbance = 0

        if plateau >= plateau_tolerance and learning_rate > min_lr:
         
          learning_rate *= 0.5
          if verbose:
            print(f'update: {update}, plateau: {plateau}, learning_rate: {learning_rate}')
          plateau = 0
          
            

        elif disturbance >= plateau_tolerance/2 and learning_rate < max_lr:
          learning_rate *= 1.5
          if verbose:
            print(f'update: {update},  disturbance: {disturbance}, learning_rate: {learning_rate}')
          disturbance = 0
        

        current_interaction += 1
       
       
        if current_interaction >= epochs:         
          break
       
        error += int(update != 0)
      self.errors.append(error)

    print(f'final learning_rate: {learning_rate}, plateau: {plateau}, disturbance: {disturbance}')
    if plateau > len(inputs):
        print(f'did not converge after {current_interaction} updates')
    else:
        print(f'converged after {current_interaction} iterations!')
    return self
  
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
  


  def decision_function(self, x):
    linear_output = np.dot(x, self.weights) + self.bias# x * w0, missing bias
   
    return _unit_step_func(linear_output)

 
