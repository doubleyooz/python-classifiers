import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, matthews_corrcoef


from models.Model import ModelInterface
from models.MaxNaiveBayes import MaxNaiveBayes
from utils import get_grid_values
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species', 'Predicted Species']

def use_classifier (data2, classifier: ModelInterface, given_point=None, old_entries=False, decision_boundary=True):
   
    modified_df = data2.copy()

    print(f'DATA SIZE: {len(modified_df)}')
    print(modified_df.head(10))
    print(modified_df.tail(10))
    print(classifier.columns)

    # Create a dictionary to map class names to numerical values
    class_mapping = {classifier.pairs[0]: 0, classifier.pairs[1]: 1}
   
    # Calculate the distance for each point on the grid   
    X_test = modified_df.iloc[:, :-1].values  # Features (all columns except the last one)
    Y_test = modified_df.iloc[:, -1].values   # Labels (last column)
   
    # Calculate the distance for each point on the grid
    fit = getattr(classifier, "fit", None)
    if callable(fit): 
        classifier.fit(modified_df)

    # modified_df[columns[5]] = classifier._naive_bayes_gaussian(X=X_test)
    modified_df[columns[5]] = modified_df.apply(classifier.classify, axis=1)
    modified_df[columns[5]] = modified_df[columns[5]].map(class_mapping)
   
  
    # Apply the mapping to Y_test
    Y_test = [class_mapping[class_name] for class_name in Y_test]
    
    print(modified_df.head(10))
   
    print(confusion_matrix(Y_test, modified_df[columns[5]]))
    print(f1_score(Y_test, modified_df[columns[5]]))
  
  
    # Plot the data points
    entries_index = 4 if old_entries else 5
    class1_df = modified_df[modified_df[columns[entries_index]] == 0]
    class2_df = modified_df[modified_df[columns[entries_index]] == 1]

   
   
    # Calculate the decision function value for each point on the grid   
    decision_equation = classifier.get_equation()

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(class1_df[columns[0]], class1_df[columns[1]], c='blue')
    plt.scatter(class2_df[columns[0]], class2_df[columns[1]], c='green')

    values_grid = get_grid_values(modified_df, columns=classifier.columns)
    
    legend = [classifier.pairs[0], classifier.pairs[1], 'Decision Boundary']

    if given_point:
        predicted_species = classifier.predict(given_point)
        plt.scatter(given_point['x1'], given_point['x2'], c='red')  # Plot the given point
        legend.insert( 2, f'Given Point {predicted_species}')

  
    if decision_boundary:       
        decision_function_values = classifier.get_decision_values(values_grid)
        plt.contour(values_grid['x1'], values_grid['x2'], decision_function_values, levels=[0], colors='purple')

    # Customize the plot
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.suptitle('Scatter Plot with Decision Boundary: Sepal Length vs. Sepal Width')

    plt.title(decision_equation, fontsize=10)
    plt.legend(legend)

    if len(classifier.columns) >= 3:
    # Start a new plot for Petal Length vs. Petal Width
        plt.figure(figsize=(8, 6))
        plt.scatter(class1_df[columns[2]], class1_df[columns[3]], c='blue')
        plt.scatter(class2_df[columns[2]], class2_df[columns[3]], c='green')
        if given_point:
            predicted_species = classifier.predict(given_point)
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



def plot_cm(data2, classifier):
    data_df = data2.copy()

    #guarantees we have only two values
    data_df.loc[data_df[columns[4]] != classifier.pairs[0], columns[4]] = classifier.pairs[1]
    print(len(data_df))
    
    # Apply the classifier to the test set
    data_df['Predicted Species'] = data_df.apply(classifier.classify, axis=1)
    mismatches = data_df[data_df['Species'] != data_df['Predicted Species']]

    print(f'Errors found: {len(mismatches)}')
    if(mismatches.size > 0):
        print(mismatches)

    # Create a confusion matrix
    cm = confusion_matrix(data_df['Species'], data_df['Predicted Species'], labels=classifier.pairs)

    # Convert the confusion matrix to a DataFrame for easier plotting
    cm_df = pd.DataFrame(cm, index=classifier.pairs, columns=classifier.pairs)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    f1_macro = f1_score(data_df['Species'], data_df['Predicted Species'], average='macro')
    f1_micro = f1_score(data_df['Species'], data_df['Predicted Species'], average='micro')
    f1_weighted = f1_score(data_df['Species'], data_df['Predicted Species'], average='weighted')
    print(f1_macro, f1_micro, f1_weighted)


def print_metrics (data2, classifier):
    data_df = data2.copy()

    #guarantees we have only two values
    data_df.loc[data_df[columns[4]] != classifier.pairs[0], columns[4]] = classifier.pairs[1]
    print(len(data_df))
    
    # Apply the classifier to the test set
    data_df[columns[5]] = data_df.apply(classifier.classify, axis=1)
    mismatches = data_df[data_df['Species'] != data_df[columns[5]]]

    print(f'Errors found: {len(mismatches)}')
    if(mismatches.size > 0):
        print(mismatches)

    Y_test =  data_df[columns[4]].to_list()
    Y_pred =  data_df[columns[5]].to_list()
    # F1-score
    f1 = f1_score(Y_test, Y_pred)
    print(f"F1-score: {f1:.4f}")

    # Cohen's Kappa
    kappa = cohen_kappa_score(Y_test, Y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")

    # Matthews correlation coefficient
    matthews = matthews_corrcoef(Y_test, Y_pred)
    print(f"Matthews Correlation Coefficient: {matthews:.4f}")
