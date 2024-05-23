
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from models.MinimumDistance4 import MinimumDistance4
from models.MinimumDistance2 import MinimumDistance2
from models.Perceptron import Perceptron
from prepareData import get_pairs, setosa_avg, versicolor_avg, virginica_avg
from test import use_classifier, plot_cm


def use_perceptron(data2):
  # Data retrieval and preperation.
  data_temp = data2.copy()
  x = data_temp.iloc[0:100, [0, 1, 2, 3]].values
  plt.scatter(x[:50, 0], x[:50, 1], color='red')
  plt.scatter(x[50:100, 0], x[50:100, 1], color='blue')
  plt.scatter(x[100:150, 0], x[100:150, 1], color='yellow')
  plt.show()

  data_temp = data_temp.iloc[0:100, 4].values
  data_temp = np.where(data_temp == 'setosa', -1, 1)

  # Model training and evaluation.
  
  p1.fit(x, data_temp)
  plt.plot(range(1, len(p1.errors) + 1), p1.errors, marker='o')
  plt.xlabel('Epochs')
  plt.ylabel('Number of misclassifications')
  plt.show()
  print(x.shape) # prints (100, 4)
  # x = x[:, :2]
  # plot_decision_regions(x, data_temp, p1=p1)
  # Showing the final results of the perceptron model.
  # plt.show()


point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}

'''
data, test = get_pairs(exclude='virginica')
pairs = ['setosa', 'versicolor']
c1 = MinimumDistance(class1_avg=setosa_avg, class2_avg=versicolor_avg)
use_classifier(data, c1, pairs=pairs, given_point=point)
use_classifier(test, c1, pairs=pairs, given_point=point)
plot_cm(test, c1, pairs=pairs)
'''
'''
data, test = get_pairs(exclude='versicolor')
pairs = ['virginica', 'setosa']
c1 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, pairs=pairs)
use_classifier(data, c1, pairs=pairs, given_point=point)
use_classifier(test, c1, pairs=pairs, given_point=point)
plot_cm(test, c1, pairs=pairs)

'''
data, test = get_pairs(exclude='setosa')
pairs = ['virginica', 'versicolor']
# c1 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, pairs=pairs)
# c2 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, pairs=pairs)
p1 = Perceptron(learning_rate=0.01, max_iters=1200, pairs=pairs)

p1.fit(test)
print(p1.weights)
use_classifier(test, p1, pairs=pairs, given_point=point, old_entries=True)
plot_cm(test, p1, pairs=pairs)
