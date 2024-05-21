import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models.Model import ModelInterface
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species', 'Predicted Species']
def use_classifier (data2, classifier: ModelInterface, pairs, given_point=None):
   

    modified_df = data2.copy()
    original_df = data2.copy()
    original_df[columns[4]] = original_df[columns[4]].map({pairs[0]: 0, pairs[1]: 1})

    # Calculate the distance for each point on the grid
    modified_df[columns[5]] = modified_df.apply(classifier.classify, axis=1)

    modified_df[columns[4]] = modified_df[columns[4]].map({pairs[0]: 0, pairs[1]: 1})
    modified_df[columns[5]] = modified_df[columns[5]].map({pairs[0]: 0, pairs[1]: 1})
    print(modified_df)
  
    # Plot the data points
    class1_df = modified_df[modified_df[columns[5]] == 0]
    class2_df = modified_df[modified_df[columns[5]] == 1]

    # Generate a grid over the feature space
    values_grid = classifier.get_grid_values(modified_df, columns)
    decision_function_values = classifier.get_decision_values(values_grid, columns)
    # print(values_grid)
    # Calculate the decision function value for each point on the grid
    decision_equation = classifier.get_equation()

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(class1_df[columns[0]], class1_df[columns[1]], c='blue')
    plt.scatter(class2_df[columns[0]], class2_df[columns[1]], c='green')

  
    legend = [pairs[0], pairs[1], 'Decision Boundary']

    if given_point:
        predicted_species = classifier.predict(given_point)
        plt.scatter(given_point['x1'], given_point['x2'], c='red')  # Plot the given point
        legend.insert( 2, f'Given Point {predicted_species}')

    #plt.contour(X, Y, decision_function_values, levels=[0], colors='purple')

    plt.contour(values_grid['x1'], values_grid['x2'], decision_function_values, levels=[0], colors='purple')

    # Customize the plot
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.suptitle('Scatter Plot with Decision Boundary: Sepal Length vs. Sepal Width')

    plt.title(decision_equation, fontsize=10)
    plt.legend(legend)

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
    plt.contour(values_grid['x3'], values_grid['x4'], decision_function_values, levels=[0], colors='purple')
    plt.title(decision_equation, fontsize=10)
    plt.legend(legend)
  
    # Show the plot
    plt.show()

def plot_cm(data2, classifier, pairs = ['setosa', 'versicolor']):
    data_df = data2.copy()

    # Apply the classifier to the test set
    data_df['Predicted Species'] = data_df.apply(classifier.classify, axis=1)
    mismatches = data_df[data_df['Species'] != data_df['Predicted Species']]

    print(f'Errors found: {mismatches.size}')
    if(mismatches.size > 0):
        print(mismatches)

    # Create a confusion matrix
    cm = confusion_matrix(data_df['Species'], data_df['Predicted Species'], labels=pairs)

    # Convert the confusion matrix to a DataFrame for easier plotting
    cm_df = pd.DataFrame(cm, index=pairs, columns=pairs)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()