
from models.BackPropagation import BackPropagation
from models.KMeans import KMeans
from models.MinimumDistance import MinimumDistance
from models.MinimumDistance2 import MinimumDistance2
from models.MinimumDistance4 import MinimumDistance4
from models.NaiveBayes import NaiveBayes
from models.Perceptron import Perceptron

models = {'MinimumDistance4': MinimumDistance4,
          'MinimumDistance2': MinimumDistance2,
          'MinimumDistance': MinimumDistance,
          'Perceptron': Perceptron,
          'KMeans': KMeans,
          'NaiveBayes': NaiveBayes,
          'BackPropagation': BackPropagation, }
