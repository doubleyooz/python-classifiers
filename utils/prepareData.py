import ast
import itertools
import pandas as pd
import numpy as np
import random


from models.MinimumDistance4 import MinimumDistance4


def get_averages(data_df=None, features=['Sepal length', 'Sepal width', 'Petal length', 'Petal width']):
    if data_df is None:
        raise ValueError(f"data_df must be defined")
    averages = {}
    averages_array = []
    for feature in features:
        averages[feature] = data_df[feature].mean()
        averages_array.append(data_df[feature].mean())
    return averages, averages_array


def get_pairs(df, class_column='Species', number_of_classes=2):

    print(f'class_column={class_column}, number_of_classes={
          number_of_classes}')

    possible_classes = df[class_column].unique().tolist()
    print(possible_classes[:10])
    for item in possible_classes:
        try:
            _ = ast.literal_eval(item)

        except (ValueError, SyntaxError):
            continue
        else:
            print('its not a string')
            return {}
    possible_classes = [str(
        item) if item is not None else 'None' for item in df[class_column].unique().tolist()]

    print('prior sorting')
    print(possible_classes[:30])
    if (len(possible_classes) > 30):
        print('too many classes')
        return {}
    classes = sorted(possible_classes)

    permutations = list(itertools.combinations(classes, number_of_classes))
    opposite_permutations = list((cl, 'non_' + cl) for cl in classes)

    temp = {}
    for permutation in permutations:
        temp[permutation[0] + ' - ' + permutation[1]] = {
            'classes': list(permutation),
            'exclude': [element for element in classes if element not in permutation][0]
        }

    for permutation in opposite_permutations:
        temp[permutation[0] + ' - ' + permutation[1]] = {
            'classes': list(permutation),
            'exclude': permutation[0]
        }

    return temp


def get_classes(df, exclude='virginica', column='Species', classes=['virginica', 'setosa', 'versicolor'], generate_numbers=False, seed=None, columns_ignored=-1, overwrite_classes=False, frac=0.7):

    print(f'exclude={exclude}')
    if generate_numbers:
        samples = 100

        random_values = {col: np.random.uniform(
            0.5, 10, samples) for col in list(df.columns[:columns_ignored])}

        # Create the DataFrame
        data_df = pd.DataFrame(random_values)
        features = list(df.columns[: columns_ignored])
        classes_copy = classes
        excluded_index = classes.index(exclude)
        classes_copy.remove(exclude)
        _, avg_list = get_averages(df, features=features)
        del avg_list[excluded_index]

        c1 = MinimumDistance4(
            class1_avg=avg_list[0], class2_avg=avg_list[1], classes=classes_copy)
        data_df[column] = data_df.apply(c1.classify, axis=1)
    else:
        print(exclude, classes)
        data_df = df[df[column] != exclude]
        opposite_data_df = df[df[column] == exclude]

        class1_df = df[df[column] == classes[0]]
        class2_df = df[df[column] != classes[0]]

    if seed is None:
        seed = random.randint(1, 1000)

    # Randomly sample 70% of your dataframe
    data = data_df.copy().sample(frac=frac, random_state=seed)

    # Get the remaining 30% of the data
    test = data_df.copy().drop(data.index)

    if overwrite_classes:
        data.loc[data[column] != classes[0], column] = classes[1]
        test.loc[test[column] != classes[0], column] = classes[1]
        class1_df.loc[class1_df[column] != classes[0], column] = classes[1]
        class2_df.loc[class2_df[column] != classes[0], column] = classes[1]
        opposite_data_df.loc[opposite_data_df[column]
                             != classes[0], column] = classes[1]

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


def get_points(labels, min=0, max=10, samples=100):
    return {col: np.random.uniform(min, max, samples) for col in list(labels)}


def load_csv(filepath):
    # Open a file dialog to select a CSV file

    if not filepath:
        return None
    try:
        # Read the CSV file into a DataFrame
        return pd.read_csv(filepath)

    except pd.errors.EmptyDataError:
        print("The selected file is empty.")
    except pd.errors.ParserError:
        print("Error parsing the CSV file. Please check the file format.")
