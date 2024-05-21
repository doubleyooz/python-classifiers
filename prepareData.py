import pandas as pd
from sklearn.metrics import confusion_matrix
    
path = './assets/Iris.csv'
df = pd.read_csv(path)
non_virginica_df = df[df['Species'] != 'virginica']
non_setosa_df = df[df['Species'] != 'setosa']
non_versicolor_df = df[df['Species'] != 'versicolor']

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


setosa_avg = [setosa_avg_x1, setosa_avg_x2, setosa_avg_x3, setosa_avg_x4]
versicolor_avg = [versicolor_avg_x1, versicolor_avg_x2, versicolor_avg_x3, versicolor_avg_x4]
virginica_avg = [virginica_avg_x1, virginica_avg_x2, virginica_avg_x3, virginica_avg_x4]

# Randomly sample 70% of your dataframe
data = non_versicolor_df.copy().sample(frac=0.7)

# Get the remaining 30% of the data
test = non_versicolor_df.copy().drop(data.index)

print(f'Training data size: {data.size}')
print(f'Testing data size: {test.size}')
# Transform the 'Species' column to 0 or 1

data = non_versicolor_df.copy()
print(f'non_versicolor_df size: {non_versicolor_df.size}')

def get_pairs(exclude):
    if exclude == 'virginica':
        data_df = non_virginica_df
    elif exclude == 'setosa':
        data_df = non_setosa_df
    elif exclude == 'versicolor':
        data_df = non_versicolor_df
    else:
        raise ValueError(f"Invalid exclude value: {exclude}")

    # Randomly sample 70% of your dataframe
    data = data_df.copy().sample(frac=0.7)

    # Get the remaining 30% of the data
    test = data_df.copy().drop(data.index)

    return data, test



# petal width and petal length are basically the same so I taking one of them out
data = data[['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']]
test = test[['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']]

data.tail(6)
