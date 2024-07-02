import numpy as np

from models.Model import ModelInterface
from utils.prepareData import get_averages

class MinimumDistance(ModelInterface):
    def __init__(self, df, class_column='Species', columns_ignored=-1):
        df_copy = df.copy()
        
        self.columns = list(df_copy.columns[: columns_ignored])
        self.classes = sorted(list(df_copy[class_column].unique()))
        print(self.classes)
        self.class_mapping = {class_label: idx for idx, class_label in enumerate(self.classes)}
        self.point_names = ['x' + str(i) for i in range(1, len(self.columns) + 1)]
      
        self.averages_dict = {}
        self.averages_list = []

        for class_label in self.classes:
            avg_dict, avg_list = get_averages(df_copy[df_copy[class_column] == class_label])
            self.averages_dict[class_label] = avg_dict
            self.averages_list.append(avg_list)

        self.metrics = {
        'fscore': 0,
        'kappa': 0,
        'matthews': 0,
        'precision': 0,
        'accuracy': 0,
        'recall': 0
        }

    def decision_function(self, row):
        x = np.array(row)                     
        class_avgs = []
        class_w0s = []
        print(self.classes)
        print(self.averages_list)
        for idx, class_label in enumerate(self.classes):
            class_avg = np.array(self.averages_list[idx])
            print(class_avg)
            class_avgs.append(class_avg)
            class_w0s.append(np.sum(class_avg**2))
        

        d1 = np.dot(x, class_avgs[0]) - (class_w0s[0] / 2)
        d2 = np.dot(x, class_avgs[1]) - (class_w0s[1] / 2)

        return d1 - d2

    def classify(self, row):
        # print(self.classes[0] if self.decision_function(row) > 0 else self.classes[1])
        return self.classes[0] if self.decision_function(row) > 0 else self.classes[1]

    def surface(self, row):
        x = np.array(row)                     
        class_avgs = []
        class_w0s = []

        for class_label in self.classes:
            class_avg = np.array(self.averages_dict[class_label])
            class_avgs.append(class_avg)
            class_w0s.append(np.sum(class_avg**2))
        

        d1 = np.dot(x, class_avgs[0]) - (class_w0s[0] / 2)
        d2 = np.dot(x, class_avgs[1]) - (class_w0s[1] / 2)
        return (d1 + d2) / 2

    def predict(self, point):
        row = {col: point[point_name] for col, point_name in zip(self.columns, self.point_names)}
        return self.classify(row)


  
    def get_equation(self):
        equation = 'Decision Boundary Equation: '
        class1_avg = self.averages_list[0]
        class2_avg = self.averages_list[1]
        for i, avg in enumerate(class1_avg):
            
            equation += f'x{i} * {round(avg, 2)}'
            if i < len(self.averages_list[0]) - 1:
                equation += ' + '

        squared_sum = sum(x ** 2 for x in self.averages_list[1]) / len(self.averages_list[1])
        result = round(squared_sum, 2)  # You can adjust the decimal places as needed

       
        equation += f' - {result}'
        return equation

