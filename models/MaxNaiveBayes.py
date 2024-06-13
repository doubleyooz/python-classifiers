import numpy as np
from models.Model import ModelInterface

class MaxNaiveBayes(ModelInterface):
    def __init__(self,  df, class_column='Species', columns_ignored=-1, gaussian=True):
       
        # the final column must refer to the classes
        df_copy = df.copy()
        self.df = {}
        self.columns = list(df_copy.columns[: columns_ignored])
        self.class_column = class_column
        self.point_names = ['x' + str(i) for i in range(1, len(self.columns) + 1)]
        self.classes = sorted(list(df_copy[class_column].unique()))
        self.means = {cl: {feat_name: None for feat_name in self.columns} for cl in self.classes}
        self.gaussian = gaussian
        self.class_mapping = {class_label: idx for idx, class_label in enumerate(self.classes)}
        
        self.std = {cl: {feat_name: None for feat_name in self.columns} for cl in self.classes}
        # number_of_classes = len(class_labels)
        self.prior = []
    
        for cl in self.classes:
            df_class = df_copy[df_copy[class_column] == cl] # it extracts all the datapoints where the Y value is the given cl
            self.prior.append(len(df_class)/len(df_copy)) # calculate the prior probability for each class. We're dividing the number of samples where Y = y by the total of samples
            self.df[cl] = df_class
            for feat_name in self.columns:             
                self.means[cl][feat_name] = df_class[feat_name].mean()
                self.std[cl][feat_name] = df_class[feat_name].std()
       
        self.errors = []
        self.metrics = {
          'fscore': 0,
          'kappa': 0,
          'matthews': 0,
          'precision': 0,
          'accuracy': 0,
          'recall': 0
        }

  
    def classify(self, row): 
        return self.classes[0] if np.argmax(self.decision_function(row)) < 1 else self.classes[1]


    def get_equation(self):
       
        str = 'yet to add the equation'

        return f"Decision Boundary Equation: {str}"


    def predict(self, point):
        row = {col: point[point_name] for col, point_name in zip(self.columns, self.point_names)}
        return self.classify(row)

    '''
    Calculating the decision boundary for a Naive Bayes classifier, especially when dealing with Gaussian-distributed features,
    involves understanding how the classifier makes decisions based on the posterior probabilities;
    '''
    def get_decision_values(self, grid):
        result = [(np.argmax(self.decision_function({self.columns[0]: x1, self.columns[1]: x2, self.columns[2]: x3, self.columns[3]: x4}))) for x1, x2, x3, x4 in zip(*[np.ravel(grid[name]) for name in self.point_names])]

        values = np.array(result)
        array_2d = values.reshape((100, 100))
        return array_2d

    def decision_function(self, x):
       
        # calculate likelihood    
        number_of_classes = len(self.classes)   
   
        likelihood = [1]* number_of_classes

        for j in range(number_of_classes): # loop over every class

            for i in self.columns:   # loop over every feature
                # multiply individual conditional probabilities to get the likelihood
                # print(i, x[i], self.classes[j])                    
                                                                    # feature_name,   val, 'class'   
                if self.gaussian:
                    likelihood[j] *= self._calculate_likelihood_gaussian(i, x[i], self.classes[j])
                else:
                    likelihood[j] *= self.calculate_likelihood_categorically(i, x[i], self.classes[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*number_of_classes

        # loop over all classes
        for j in range(number_of_classes):
            post_prob[j] = likelihood[j] * self.prior[j]
    
        return post_prob
    
    
    def calculate_likelihood_categorically(self, feat_name, feat_value,  class_value):       
        df_class = self.df[class_value]
        p_x_given_y = len(df_class[df_class[feat_name]==feat_value]) / len(df_class)
        return p_x_given_y

    def _calculate_likelihood_gaussian(self, feat_name, feat_val, class_label):     
      
        mean = self.means[class_label][feat_name]
        std = self.std[class_label][feat_name]
        p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std) * np.exp(-((feat_val-mean)**2) / (2 * std**2))) # normal distribution
        return p_x_given_y
    
    def _naive_bayes_gaussian(self, X):
       
        Y_pred = []              

        #loop over every data sample    
        for x in X:
            # calculate likelihood                
            x_dict = {}
            if(type(x) is np.ndarray):
                for i in range(len(self.columns)):
                    x_dict[self.columns[i]] = x[i]
            else:
                x_dict = x
            post_prob = self.decision_function(x_dict)
            Y_pred.append(np.argmax(post_prob)) # add to the returning array the class where the posterior probability is the highest

        return np.array(Y_pred)
    