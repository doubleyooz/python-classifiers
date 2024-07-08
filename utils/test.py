import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, matthews_corrcoef


from models.Model import ModelInterface

from utils.npHelper import get_decision_values, get_grid_values
from utils.metrics import accuracy, f1, precision, recall
columns = ['Sepal length', 'Sepal width', 'Petal length',
           'Petal width', 'Species', 'Predicted Species']


def extract_values(dataframe, skip_columns=-1):
    # Features (all columns except the last one)
    X_test = dataframe.iloc[:, :skip_columns].values
    Y_test = dataframe.iloc[:, skip_columns].values   # Labels (last column)

    return X_test, Y_test


def extract_y_test_y_pred(dataframe, model, pred_col="Prediction", skip_columns=-1, map_values=False):
    X_test, Y_test = extract_values(dataframe, skip_columns)
    dataframe[pred_col] = dataframe.iloc[:, :-1].apply(model.classify, axis=1)

    if map_values:
        dataframe[pred_col] = dataframe[pred_col].map(model.class_mapping)

    return X_test, Y_test, dataframe[pred_col]


def map_values(values, class_mapping, reverse=False):
    if reverse:
        {v: k for k, v in class_mapping.items()}
    return [class_mapping[class_name] for class_name in values]


def use_classifier(data2, model: ModelInterface, given_point=None, old_entries=False, decision_boundary=True, fit=False):

    modified_df = data2.copy()

    print(f'DATA SIZE: {len(modified_df)}', model.columns)
    print(modified_df.head(10))
    print(modified_df.tail(10))

    # Create a dictionary to map class names to numerical values

    X_test, Y_test = extract_values(modified_df)
    print(Y_test)
    print(model.class_mapping)
    Y_test = map_values(Y_test, model.class_mapping)
    # Calculate the distance for each point on the grid
    has_fit = getattr(model, "fit", None)
    if callable(has_fit) and fit:
        model.fit(inputs=X_test, targets=Y_test,
                  learning_rate=0.01, epochs=1000)

    _, _, Y_pred = extract_y_test_y_pred(
        dataframe=modified_df, model=model, pred_col=columns[5], map_values=True)

    print(modified_df.head(10))

    print(confusion_matrix(Y_test, Y_pred))

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
        plt.scatter(given_point['x1'], given_point['x2'],
                    c='red')  # Plot the given point
        legend.insert(2, f'Given Point {predicted_species}')

    print(f"decision_boundary: {decision_boundary}")
    print(f"point_names: {model.point_names}")
    '''
    [0.02759677 0.02833139 0.02916404 ... 0.98531765 0.98533118 0.98534353]
 [0.02745206 0.02816882 0.02898079 ... 0.98531468 0.98532846 0.98534105]
    '''
    if decision_boundary:
        # decision_function_values = model.get_decision_values(values_grid)
        decision_function_values = get_decision_values(
            values_grid, model.point_names, model.decision_function)
        print(f"decision_function_values: {decision_function_values}")
        plt.contour(values_grid['x1'], values_grid['x2'],
                    decision_function_values, levels=[0], colors='purple')

    # Customize the plot
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.suptitle(
        'Scatter Plot with Decision Boundary: Sepal Length vs. Sepal Width')

    plt.title(decision_equation, fontsize=10)
    plt.legend(legend)

    if len(model.columns) >= 3:
        # Start a new plot for Petal Length vs. Petal Width
        plt.figure(figsize=(8, 6))
        plt.scatter(class1_df[columns[2]], class1_df[columns[3]], c='blue')
        plt.scatter(class2_df[columns[2]], class2_df[columns[3]], c='green')
        if given_point:
            predicted_species = model.predict(given_point)
            # Plot the given point
            plt.scatter(given_point['x3'], given_point['x4'], c='red')
            legend.insert(2, f'Given Point {predicted_species}')

        # Customize the plot
        plt.xlabel(columns[2])
        plt.ylabel(columns[3])
        plt.suptitle(
            'Scatter Plot with Decision Boundary: Petal Length vs. Petal Width')
        if decision_boundary:
            plt.contour(values_grid['x3'], values_grid['x4'],
                        decision_function_values, levels=[0], colors='purple')
        plt.title(decision_equation, fontsize=10)
        plt.legend(legend)

    # Show the plot
    plt.show()


def plot_cm(data2, model):
    data_df = data2.copy()

    # guarantees we have only two values
    data_df.loc[data_df[columns[4]] !=
                model.classes[0], columns[4]] = model.classes[1]
    print(len(data_df))
    print(data_df.head(5))

    x_test, y_test, y_pred = extract_y_test_y_pred(
        dataframe=data_df, model=model, pred_col=columns[5])

    # Apply the model to the test set

    mismatches = data_df[data_df[columns[4]] != data_df[columns[5]]]

    print(f'Errors found: {len(mismatches)}')
    if (mismatches.size > 0):
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


def calculate_metrics(dataframe, model):
    data_df = dataframe.copy()

    x_test, y_test, y_pred = extract_y_test_y_pred(
        dataframe=data_df, model=model, pred_col=columns[5], map_values=True)

    y_test = y_test.tolist()
    y_pred = y_pred.tolist()

    y_test = map_values(y_test, model.class_mapping)

    model.metrics['fscore'] = f1(y_test, y_pred)
    model.metrics['precision'] = precision(y_test, y_pred)
    model.metrics['recall'] = recall(y_test, y_pred)

    model.metrics['accuracy'] = accuracy(y_test, y_pred)
    model.metrics['kappa'] = cohen_kappa_score(y_test, y_pred)
    model.metrics['matthews'] = matthews_corrcoef(y_test, y_pred)


def print_metrics(model):
    print(f"F1-score: {model.metrics['fscore']:.4f}")
    print(f"precision: {model.metrics['precision']:.4f}")
    print(f"recall: {model.metrics['recall']:.4f}")

    print(f"accuracy: {model.metrics['accuracy']:.4f}")
    print(f"Cohen's Kappa: {model.metrics['kappa']:.4f}")
    print(f"Matthews Correlation Coefficient: {model.metrics['matthews']:.4f}")
