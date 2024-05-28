import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix

from models.MinimumDistance4 import MinimumDistance4
    
path = './assets/Iris.csv'
df = pd.read_csv(path)
non_virginica_df = df[df['Species'] != 'virginica']
non_setosa_df = df[df['Species'] != 'setosa']
non_versicolor_df = df[df['Species'] != 'versicolor']

virginica_df = df[df['Species'] == 'virginica']
setosa_df = df[df['Species'] == 'setosa']
versicolor_df = df[df['Species'] == 'versicolor']

# Select the desired columns
setosa_df = df[df['Species'] == 'setosa']
versicolor_df = df[df['Species'] == 'versicolor']
virginica_df = df[df['Species'] == 'virginica']


setosa_avg_x1 = setosa_df['Sepal length'].mean()
setosa_avg_x2 = setosa_df['Sepal width'].mean()
setosa_avg_x3 = setosa_df['Petal length'].mean()
setosa_avg_x4 = setosa_df['Petal width'].mean()

versicolor_avg_x1 = versicolor_df['Sepal length'].mean()
versicolor_avg_x2 = versicolor_df['Sepal width'].mean()
versicolor_avg_x3 = versicolor_df['Petal length'].mean()
versicolor_avg_x4 = versicolor_df['Petal width'].mean()

virginica_avg_x1 = virginica_df['Sepal length'].mean()
virginica_avg_x2 = virginica_df['Sepal width'].mean()
virginica_avg_x3 = virginica_df['Petal length'].mean()
virginica_avg_x4 = virginica_df['Petal width'].mean()

setosa_avg_x1 = setosa_df['Sepal length'].mean()
setosa_avg_x2 = setosa_df['Sepal width'].mean()
setosa_avg_x3 = setosa_df['Petal length'].mean()
setosa_avg_x4 = setosa_df['Petal width'].mean()


setosa_avg = [setosa_avg_x1, setosa_avg_x2, setosa_avg_x3, setosa_avg_x4]
versicolor_avg = [versicolor_avg_x1, versicolor_avg_x2, versicolor_avg_x3, versicolor_avg_x4]
virginica_avg = [virginica_avg_x1, virginica_avg_x2, virginica_avg_x3, virginica_avg_x4]

# Randomly sample 70% of your dataframe
data = non_versicolor_df.copy().sample(frac=0.7)

# Get the remaining 30% of the data
test = non_versicolor_df.copy().drop(data.index)

def get_averages(data_df = None, features=['Sepal length', 'Sepal width', 'Petal length', 'Petal width']):
    if data_df is None:
        raise ValueError(f"data_df must be defined")
    averages = {}
    averages_array = []
    for feature in features:
        averages[feature] = data_df[feature].mean()
        averages_array.append(data_df[feature].mean())
    return averages, averages_array

def get_pairs(exclude = 'virginica', column='Species', pairs = ['virginica', 'setosa', 'versicolor'], generate_numbers = False, seed=None, overwrite_classes=False):
    
    print(f'exclude={exclude}')
    if generate_numbers:
        samples = 100
        random_values = {
            'Sepal length': np.random.uniform(0.5, 10, samples),
            'Sepal width': np.random.uniform(0.5, 10, samples),
            'Petal length': np.random.uniform(0.5, 10, samples),
            'Petal width': np.random.uniform(0.5, 10, samples)
        }

        # Create the DataFrame
        data_df = pd.DataFrame(random_values)       
      

        pairs_copy = pairs
        excluded_index = pairs.index(exclude)
        pairs_copy.remove(exclude)
        avg_list = [virginica_avg, setosa_avg, versicolor_avg ]
        del avg_list[excluded_index]

        c1 = MinimumDistance4(class1_avg=avg_list[0], class2_avg=avg_list[1], pairs=pairs_copy)
        data_df[column] = data_df.apply(c1.classify, axis=1)
    else: 
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



def load_csv(self, filepath):
    # Open a file dialog to select a CSV file
   
    if filepath:
        try:
            # Read the CSV file into a DataFrame
            return pd.read_csv(filepath)
          
        except pd.errors.EmptyDataError:
            print("The selected file is empty.")
        except pd.errors.ParserError:
            print("Error parsing the CSV file. Please check the file format.")
        finally:
            return None


# petal width and petal length are basically the same so I taking one of them out
data = data[['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']]
test = test[['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']]

data.tail(6)
