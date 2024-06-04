import curses 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from models.BackPropagation import NeuralNetwork
from models.MinimumDistance4 import MinimumDistance4
from models.MinimumDistance2 import MinimumDistance2
from models.Perceptron import Perceptron
from models.MaxNaiveBayes import MaxNaiveBayes
from prepareData import get_pairs, get_averages
from test import use_classifier, plot_cm, print_metrics

point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}

'''
data, test = get_pairs(exclude='virginica')
pairs = ['setosa', 'versicolor']
c1 = MinimumDistance(class1_avg=setosa_avg, class2_avg=versicolor_avg)
use_classifier(data, c1,  given_point=point)
use_classifier(test, c1,  given_point=point)
plot_cm(test, c1, pairs=pairs)
'''
'''
data, test = get_pairs(exclude='versicolor')
pairs = ['virginica', 'setosa']
c1 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, pairs=pairs)
use_classifier(data, c1,  given_point=point)
use_classifier(test, c1,  given_point=point)
plot_cm(test, c1, pairs=pairs)

'''

'''
data, test = get_pairs(exclude='setosa')
pairs = ['virginica', 'versicolor']
# c1 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, pairs=pairs)
# c2 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, pairs=pairs)
p1 = Perceptron(learning_rate=0.01, max_iters=1200, pairs=pairs)

p1.fit(test)
print(p1.weights)
use_classifier(test, p1,  given_point=point, old_entries=True)
plot_cm(test, p1, pairs=pairs)

    '''
def main():
    menu = ["MinimumDistance4", "MinimumDistance2", "Perceptron", "MaxBayes", "BackPropagation"]
    dict_pairs = {
        'versicolor - setosa': {
            'exclude': 'virginica',
            'pairs': ['versicolor', 'setosa']
        },
        'versicolor - virginica': {
            'exclude': 'setosa',
            'pairs': ['virginica', 'versicolor']
        },

        'virginica - setosa': {
            'exclude': 'versicolor',
            'pairs': ['virginica', 'setosa']
        },
        
     
        'versicolor - non_versicolor': {
            'exclude': 'versicolor',
            'pairs': ['versicolor', 'non_versicolor']
        },
       

        'setosa - non_setosa': {
            'exclude': 'setosa',
            'pairs': ['setosa', 'non_setosa']
        },
        'virginica - non_virginica': {
            'exclude': 'virginica',
            'pairs': ['virginica', 'non_virginica' ]
        },
    }
    pairs_list = list(dict_pairs.keys())
    print(pairs_list)
    
    point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}
    while True:              
        selected_option = show_menu_options("Select the model to be used", menu=menu, last_option="Shut down")
        print(selected_option)
        if selected_option == len(menu):
            print('end of operation')
            break
        selected_model = menu[selected_option]

        while True:
            # Use MinimumDistance4 model
            print(selected_model)
            selected_option = show_menu_options("Select the pairs to be considered", menu=pairs_list)
            if selected_option == len(pairs_list):
                break
            selected_pair = dict_pairs[pairs_list[selected_option]]
        
            data, test, class1_df, class2_df, opposite_data_df = get_pairs(exclude=selected_pair['exclude'], pairs=selected_pair['pairs'], overwrite_classes=True)
     
            class1_avg_dict , class1_avg_list = get_averages(class1_df)
            class2_avg_dict , class2_avg_list = get_averages(class2_df)
            print(class1_avg_list)
            print(class2_avg_list)

            while True: 
                title = f'{selected_model} - {selected_pair['pairs']}'
                print(title)
                datasets_list = ['training', 'test', 'full']
                selected_option = show_menu_options("Select dataset to be used", menu=datasets_list)
                if selected_option == len(datasets_list):
                    break
                
                if selected_option == 0:
                    selected_dataset = {'dataset': data, 'title': datasets_list[selected_option]}
                elif selected_option == 1:
                    selected_dataset = {'dataset': test, 'title': datasets_list[selected_option]}
                else: 
                    selected_dataset = {'dataset': pd.concat([data, test]), 'title': datasets_list[selected_option]} 
                

                if selected_model == "Perceptron":
                    use_perceptron(selected_model=selected_model, selected_pair=selected_pair['pairs'], selected_dataset=selected_dataset, exclude=selected_pair['exclude'])

                elif any(selected_model in x for x in ["MinimumDistance4", "MinimumDistance2"]):
                    use_minimum_distance_classifier(selected_model=selected_model, selected_dataset=selected_dataset, selected_pair=selected_pair['pairs'], class1_avg_list=class1_avg_list, class2_avg_list=class2_avg_list)
              
                elif selected_model == "MaxBayes":
                    use_bayes(selected_model=selected_model, selected_dataset=selected_dataset, selected_pair=selected_pair['pairs'])
                
                elif selected_model == 'BackPropagation':
                    use_backpropagation(selected_model=selected_model, selected_dataset=selected_dataset, selected_pair=selected_pair['pairs'])
                

def show_menu_options(title, menu, last_option="Go back"):
    print(f'\n{title}')
    options_list = menu + [last_option] 
    while True:
        try:
            for idx, row in enumerate(options_list):
                print(f"{idx} - {row}")
            key = int(input("Option selected: "))
           
            if key <= len(menu) and key >= 0:
                return key
            else:
                raise ValueError
       
        except ValueError:
            print('invalid option\n')
        except KeyboardInterrupt:
            print('\nOperation canceled by user.')
            exit(0)

          
def use_minimum_distance_classifier(selected_model, selected_pair, selected_dataset, class1_avg_list, class2_avg_list):
    print(len(selected_dataset['dataset']))
  
    if selected_model == 'MinimumDistance4':
        c1 = MinimumDistance4(class1_avg=class1_avg_list, class2_avg=class2_avg_list, pairs=selected_pair)
    else:
        c1 = MinimumDistance2(class1_avg=class1_avg_list[:2], class2_avg=class2_avg_list[:2], pairs=selected_pair)

    print(f'class_1: {c1.class1_avg}; class_2: {c1.class2_avg}')
    while True:
        title = f'{selected_model} - {selected_pair} - {selected_dataset['title']}'
        print(title)
        actions_list = ['classify', 'predict_point', 'confusion_matrix']
        selected_option = show_menu_options("Select action:", menu=actions_list)
        if selected_option == len(actions_list):
            break
        
        if selected_option == 0:
            print(f'{title}: classify')
            use_classifier(selected_dataset['dataset'], c1)
        elif selected_option == 1:
            print(f'{title}: predict')
            use_classifier(selected_dataset['dataset'], c1, given_point=point)
        else:  
            print(f'{title}: confusion_matrix')                       
            plot_cm(selected_dataset['dataset'], c1)

          
def use_perceptron(selected_model, selected_pair, selected_dataset, exclude):
    
    p1 = Perceptron(learning_rate=0.01, max_iters=1200, pairs=selected_pair)

    while True:
        title = f'\n{selected_model} - {selected_pair} - {selected_dataset['title']} - {p1.weights}'
        print(title)
        actions_list = ['classify', 'fit', 'predict_point', 'confusion_matrix', 'print_metrics']
        selected_option = show_menu_options("Select action:", menu=actions_list)
        if selected_option == len(actions_list):
            break
        
        if selected_option == 0:
            print(f'{title}: classify')
            use_classifier(selected_dataset['dataset'], p1)
        elif selected_option == 1:
            print(f'{title}: fit')
            data, test, class1_df, class2_df, opposite_data_df = get_pairs(exclude=exclude, pairs=selected_pair, overwrite_classes=True)
            if selected_dataset['title'] == 'training':
                dataset = data
            elif selected_dataset['title'] == 'test':
                dataset = test
            else: 
                dataset = pd.concat([data, test])
                
            p1.fit(dataset)
        elif selected_option == 2:
            print(f'{title}: predict')
            use_classifier(selected_dataset['dataset'], p1, given_point=point)
        elif selected_option == 3:  
            print(f'{title}: confusion_matrix')                       
            plot_cm(selected_dataset['dataset'], p1)
        else:  
            print(f'{title}: predict')
            print_metrics(selected_dataset['dataset'], p1)
            



def use_bayes(selected_model, selected_pair, selected_dataset):
    print(len(selected_dataset['dataset']))
  
    bayes_1 = MaxNaiveBayes(pairs=selected_pair)

    while True:
        title = f'\n{selected_model} - {selected_pair} - {selected_dataset['title']}'
        print(title)
        bayes_1.initialise(df=selected_dataset['dataset'].copy(), Y='Species')
        actions_list = ['classify', 'predict_point', 'confusion_matrix', 'print_metrics']
        selected_option = show_menu_options("Select action:", menu=actions_list)
        if selected_option == len(actions_list):
            break
        
        if selected_option == 0:
            print(f'{title}: classify')
            use_classifier(selected_dataset['dataset'], bayes_1, decision_boundary=True)
      
        elif selected_option == 1:                
            print(f'{title}: predict')
            use_classifier(selected_dataset['dataset'], bayes_1, given_point=point, decision_boundary=True)
        elif selected_option == 2:                
            print(f'{title}: confusion_matrix')                       
            plot_cm(selected_dataset['dataset'], bayes_1)
        else:  
            print(f'{title}: predict')
            print_metrics(selected_dataset['dataset'], bayes_1)
                  


def use_backpropagation(selected_model, selected_pair, selected_dataset):
    print(len(selected_dataset['dataset']))
  
    back_prop1 = NeuralNetwork(df=selected_dataset['dataset'].copy(), class_column='Species')

    while True:
        title = f'\n{selected_model} - {selected_pair} - {selected_dataset['title']}'
        print(title)
        # back_prop1.initialise(df=selected_dataset['dataset'].copy(), Y='Species')
        actions_list = ['classify', 'predict_point', 'confusion_matrix', 'print_metrics']
        selected_option = show_menu_options("Select action:", menu=actions_list)
        if selected_option == len(actions_list):
            break
        
        if selected_option == 0:
            print(f'{title}: classify')
            # use_classifier(selected_dataset['dataset'], back_prop1, decision_boundary=True)
            
            
            modified_df = selected_dataset['dataset'].copy()
            X_test = modified_df.iloc[:, :-1].values  # Features (all columns except the last one)
            Y_test = modified_df.iloc[:, -1].values   # Labels (last column)
            for i in range(1000): #trains the NN 1000 times
                if (i % 100 == 0):
                    print(f'y_test: {Y_test}, feedForward: {back_prop1.feedForward(X_test)}')
                  
                    print("Loss: " + str(np.mean(np.square(Y_test - back_prop1.feedForward(X_test)))))
                back_prop1.train(X_test, Y_test)
                    
            print("Input: " + str(X_test))
            print("Actual Output: " + str(Y_test))
            print("Loss: " + str(np.mean(np.square(Y_test - back_prop1.feedForward(X_test)))))
            print("\n")
            print("Predicted Output: " + str(back_prop1.feedForward(X_test)))
        '''
        elif selected_option == 1:                
            print(f'{title}: predict')
            use_classifier(selected_dataset['dataset'], back_prop1, given_point=point, decision_boundary=True)
        elif selected_option == 2:                
            print(f'{title}: confusion_matrix')                       
            plot_cm(selected_dataset['dataset'], back_prop1)
        else:  
            print(f'{title}: predict')
            print_metrics(selected_dataset['dataset'], back_prop1)
        '''
  

def select_pairs(pairs_list, dict_pairs):
    selected_option = show_menu_options("Select the pairs to be considered", menu=pairs_list)
   
    selected_pair = dict_pairs[pairs_list[selected_option]]
    
    data, test, class1_df, class2_df, opposite_data_df = get_pairs(exclude=selected_pair['exclude'], pairs=selected_pair['pairs'], overwrite_classes=True)
    
    class1_avg_dict , class1_avg_list = get_averages(class1_df)
    class2_avg_dict , class2_avg_list = get_averages(class2_df)
    print(class1_avg_list)
    print(class2_avg_list)
   

if __name__ == "__main__":
    main()