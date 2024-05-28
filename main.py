import curses 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from models.MinimumDistance4 import MinimumDistance4
from models.MinimumDistance2 import MinimumDistance2
from models.Perceptron import Perceptron
from models.MaxNaiveBayes import MaxNaiveBayes
from prepareData import get_pairs, get_averages
from test import use_classifier, plot_cm, use_classifier2, plot_cm2, print_metrics

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
    menu = ["MinimumDistance4", "MinimumDistance2", "Perceptron", "MaxBayes"]
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

        if selected_model == "Perceptron":
           use_perceptron(dict_pairs=dict_pairs, pairs_list=pairs_list)

        elif selected_model == "MinimumDistance4":
            use_minimum_distance_classifier( selected_model=selected_model, dict_pairs=dict_pairs, pairs_list=pairs_list)
        elif selected_model == "MinimumDistance2":
            use_minimum_distance_classifier( selected_model=selected_model, dict_pairs=dict_pairs, pairs_list=pairs_list)
        elif selected_model == "MaxBayes":
            use_bayes(dict_pairs=dict_pairs, pairs_list=pairs_list)

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

          
def use_minimum_distance_classifier(selected_model, pairs_list, dict_pairs):
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
                dataset = data
            elif selected_option == 1:
                dataset = test
            else: 
                dataset = pd.concat([data, test])
            print(len(dataset))
            if selected_model == 'MinimumDistance4':
                c1 = MinimumDistance4(class1_avg=class1_avg_list, class2_avg=class2_avg_list, pairs=selected_pair['pairs'])
            else:
                c1 = MinimumDistance2(class1_avg=class1_avg_list[:2], class2_avg=class2_avg_list[:2], pairs=selected_pair['pairs'])
    
            print(f'class_1: {c1.class1_avg}; class_2: {c1.class2_avg}')
            while True:
                title = f'{selected_model} - {selected_pair['pairs']} - {datasets_list[selected_option]}'
                print(title)
                actions_list = ['classify', 'predict_point', 'confusion_matrix']
                selected_option = show_menu_options("Select action:", menu=actions_list)
                if selected_option == len(actions_list):
                    break
                
                if selected_option == 0:
                    print(f'{title}: classify')
                    use_classifier(dataset, c1)
                elif selected_option == 1:
                    print(f'{title}: predict')
                    use_classifier(dataset, c1, given_point=point)
                else:  
                    print(f'{title}: confusion_matrix')                       
                    plot_cm(dataset, c1)

          
def use_perceptron(pairs_list, dict_pairs):
    while True:
        # Use MinimumDistance4 model
        selected_model = 'Perceptron'
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
            selected_dataset = show_menu_options("Select dataset to be used", menu=datasets_list)
            if selected_dataset == len(datasets_list):
                break
            
            if selected_dataset == 0:
                dataset = data
            elif selected_dataset == 1:
                dataset = test
            else: 
                dataset = pd.concat([data, test])
            print(len(dataset))
            p1 = Perceptron(learning_rate=0.01, max_iters=1200, pairs=selected_pair['pairs'])


            while True:
                title = f'\n{selected_model} - {selected_pair['pairs']} - {selected_dataset} - {p1.weights}'
                print(title)
                actions_list = ['classify', 'fit', 'predict_point', 'confusion_matrix', 'print_metrics']
                selected_option = show_menu_options("Select action:", menu=actions_list)
                if selected_option == len(actions_list):
                    break
                
                if selected_option == 0:
                    print(f'{title}: classify')
                    use_classifier(dataset, p1)
                elif selected_option == 1:
                    print(f'{title}: fit')
                    selected_pair = dict_pairs[pairs_list[selected_option]]
                    data, test, class1_df, class2_df, opposite_data_df = get_pairs(exclude=selected_pair['exclude'], pairs=selected_pair['pairs'], overwrite_classes=True)
                    if selected_dataset == 0:
                        dataset = data
                    elif selected_dataset == 1:
                        dataset = test
                    else: 
                        dataset = pd.concat([data, test])
                        
                    p1.fit(dataset)
                elif selected_option == 2:
                    print(f'{title}: predict')
                    use_classifier(dataset, p1, given_point=point)
                elif selected_option == 3:  
                    print(f'{title}: confusion_matrix')                       
                    plot_cm(dataset, p1)
                else:  
                    print(f'{title}: predict')
                    print_metrics(dataset, p1)
                  



def use_bayes(pairs_list, dict_pairs):
    while True:
        # Use MinimumDistance4 model
        selected_model = 'MaxNaiveBayes'
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
            selected_dataset = show_menu_options("Select dataset to be used", menu=datasets_list)
            if selected_dataset == len(datasets_list):
                break
            
            if selected_dataset == 0:
                dataset = data
            elif selected_dataset == 1:
                dataset = test
            else: 
                dataset = pd.concat([data, test])
            print(len(dataset))
            bayes_1 = MaxNaiveBayes(learning_rate=0.01, max_iters=1200, pairs=selected_pair['pairs'])


            while True:
                title = f'\n{selected_model} - {selected_pair['pairs']} - {selected_dataset}'
                print(title)
                actions_list = ['classify', 'predict_point', 'confusion_matrix', 'print_metrics']
                selected_option = show_menu_options("Select action:", menu=actions_list)
                if selected_option == len(actions_list):
                    break
                
                if selected_option == 0:
                    print(f'{title}: classify')
                    use_classifier2(dataset, bayes_1, decision_boundary=False)
                elif selected_option == 1:                
                    print(f'{title}: predict')
                    use_classifier2(dataset, bayes_1, given_point=point, decision_boundary=False)
                elif selected_option == 2:                
                    print(f'{title}: confusion_matrix')                       
                    plot_cm2(dataset, bayes_1)
                else:  
                    print(f'{title}: predict')
                    print_metrics(dataset, bayes_1)
                  


     
   

if __name__ == "__main__":
    main()