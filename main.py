
from matplotlib import pyplot as plt
import pandas as pd

import seaborn as sns

from models.MinimumDistance import MinimumDistance
from prepareData import get_pairs, setosa_avg, versicolor_avg
from test import use_classifier

data, test = get_pairs('virginica')

point = {'x1': 4.7, 'x2': 4.4, 'x3': 1.5, 'x4': 0.4}
c1 = MinimumDistance(setosa_avg=setosa_avg, versicolor_avg=versicolor_avg)
use_classifier(data, c1, pairs=['setosa', 'versicolor'], given_point=point)
# use_classifier(test, c1, pairs=['setosa', 'versicolor'])


# Create a pairplot
# sns.pairplot(data, hue='Species')
# plt.show()