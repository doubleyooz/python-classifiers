import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, matthews_corrcoef


from models.Model import ModelInterface
from models.MaxNaiveBayes import MaxNaiveBayes
from utils import get_grid_values
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species', 'Predicted Species']

def extract_values(dataframe, skip_columns=-1):
    X_test = dataframe.iloc[:, :skip_columns].values  # Features (all columns except the last one)
    Y_test = dataframe.iloc[:, skip_columns].values   # Labels (last column)

    return X_test, Y_test
   
def map_values(values, class_mapping, reverse=False):
    if reverse:
        {v: k for k, v in class_mapping.items()}
    return [class_mapping[class_name] for class_name in values]

def use_classifier (data2, model: ModelInterface, given_point=None, old_entries=False, decision_boundary=True):
   
    modified_df = data2.copy()

    print(f'DATA SIZE: {len(modified_df)}', model.columns)
    print(modified_df.head(10))
    print(modified_df.tail(10))

    # Create a dictionary to map class names to numerical values
  
    X_test, Y_test = extract_values(modified_df)

    Y_test = map_values(Y_test, model.class_mapping)
    # Calculate the distance for each point on the grid
    fit = getattr(model, "fit", None)
    if callable(fit): 
        model.fit(inputs=X_test, targets=Y_test, learning_rate=0.01, epochs=1000)

    # modified_df[columns[5]] = model._naive_bayes_gaussian(X=X_test)
    modified_df[columns[5]] = modified_df.iloc[:, :-1].apply(model.classify, axis=1)
    modified_df[columns[5]] = modified_df[columns[5]].map(model.class_mapping)
   
    print(modified_df.head(10))
   
    print(confusion_matrix(Y_test, modified_df[columns[5]]))
    print(f1_score(Y_test, modified_df[columns[5]]))
  
  
    # Plot the data points
    entries_index = 4 if old_entries else 5
    class1_df = modified_df[modified_df[columns[entries_index]] == 0]
    class2_df = modified_df[modified_df[columns[entries_index]] == 1]

   
    # Calculate the decision function value for each point on the grid   
    decision_equation = model.get_equation()

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(class1_df[columns[0]], class1_df[columns[1]], c='blue')
    plt.scatter(class2_df[columns[0]], class2_df[columns[1]], c='green')

    values_grid = get_grid_values(modified_df, columns=model.columns)
    
    legend = [model.classes[0], model.classes[1], 'Decision Boundary']

    if given_point:
        predicted_species = model.predict(given_point)
        plt.scatter(given_point['x1'], given_point['x2'], c='red')  # Plot the given point
        legend.insert( 2, f'Given Point {predicted_species}')

  
    if decision_boundary:       
        decision_function_values = model.get_decision_values(values_grid)
        plt.contour(values_grid['x1'], values_grid['x2'], decision_function_values, levels=[0], colors='purple')

    # Customize the plot
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.suptitle('Scatter Plot with Decision Boundary: Sepal Length vs. Sepal Width')

    plt.title(decision_equation, fontsize=10)
    plt.legend(legend)

    if len(model.columns) >= 3:
    # Start a new plot for Petal Length vs. Petal Width
        plt.figure(figsize=(8, 6))
        plt.scatter(class1_df[columns[2]], class1_df[columns[3]], c='blue')
        plt.scatter(class2_df[columns[2]], class2_df[columns[3]], c='green')
        if given_point:
            predicted_species = model.predict(given_point)
            plt.scatter(given_point['x3'], given_point['x4'], c='red')  # Plot the given point
            legend.insert( 2, f'Given Point {predicted_species}')

        # Customize the plot
        plt.xlabel(columns[2])
        plt.ylabel(columns[3])
        plt.suptitle('Scatter Plot with Decision Boundary: Petal Length vs. Petal Width')
        if decision_boundary:
            plt.contour(values_grid['x3'], values_grid['x4'], decision_function_values, levels=[0], colors='purple')
        plt.title(decision_equation, fontsize=10)
        plt.legend(legend)
  
    # Show the plot
    plt.show()



def plot_cm(data2, model):
    data_df = data2.copy()

 
    
    #guarantees we have only two values
    data_df.loc[data_df[columns[4]] != model.classes[0], columns[4]] = model.classes[1]
    print(len(data_df))    
    print(data_df.head(5))

    # Apply the model to the test set   
    data_df[columns[5]] = data_df.iloc[:, :-1].apply(model.classify, axis=1)
    mismatches = data_df[data_df[columns[4]] != data_df[columns[5]]]

    
    x_test, y_test = extract_values(data_df.iloc[:, :-1])
    x_test, y_pred = extract_values(data_df)

    print(f'Errors found: {len(mismatches)}')
    if(mismatches.size > 0):
        print(mismatches)

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes)

    # Convert the confusion matrix to a DataFrame for easier plotting
    cm_df = pd.DataFrame(cm, index=model.classes, columns=model.classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
 


    '''
    'micro': Computes metrics globally by counting true positives, false negatives, and false positives across all classes.
    'macro': Calculates metrics for each label and finds their unweighted mean.
    'weighted': Computes the average weighted by the number of samples in each class.
    'binary': Only applicable for binary classification (ignores the pos_label parameter).


    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f1_macro, f1_micro, f1_weighted)
    '''

 
   
    # save metrics
    model.metrics['fscore'] = f1_score(y_test, y_pred, average='weighted')
    model.metrics['kappa'] = cohen_kappa_score(y_test, y_pred)
    model.metrics['matthews'] = matthews_corrcoef(y_test, y_pred)
    
 


def print_metrics (data2, model):
    data_df = data2.copy()

    #guarantees we have only two values
    data_df.loc[data_df[columns[4]] != model.classes[0], columns[4]] = model.classes[1]
    print(len(data_df))
    
  
  
    print(f"F1-score: {model.metrics['fscore']:.4f}")
    print(f"Cohen's Kappa: {model.metrics['kappa']:.4f}")
    print(f"Matthews Correlation Coefficient: {model.metrics['matthews']:.4f}")
  
