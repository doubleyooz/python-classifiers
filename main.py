import curses 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from models.BackPropagation import BackPropagation
from models.MinimumDistance4 import MinimumDistance4
from models.MinimumDistance2 import MinimumDistance2
from models.Perceptron import Perceptron
from models.MaxNaiveBayes import MaxNaiveBayes
from prepareData import get_classes, get_averages
from test import use_classifier, plot_cm, print_metrics, extract_values, map_values

point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}

'''
data, test = get_classes(exclude='virginica')
classes = ['setosa', 'versicolor']
c1 = MinimumDistance(class1_avg=setosa_avg, class2_avg=versicolor_avg)
use_classifier(data, c1,  given_point=point)
use_classifier(test, c1,  given_point=point)
plot_cm(test, c1, classes=classes)
'''
'''
data, test = get_classes(exclude='versicolor')
classes = ['virginica', 'setosa']
c1 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, classes=classes)
use_classifier(data, c1,  given_point=point)
use_classifier(test, c1,  given_point=point)
plot_cm(test, c1, classes=classes)

'''

'''
data, test = get_classes(exclude='setosa')
classes = ['virginica', 'versicolor']
# c1 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, classes=classes)
# c2 = MinimumDistance(class1_avg=virginica_avg, class2_avg=setosa_avg, classes=classes)
p1 = Perceptron(learning_rate=0.01, max_iters=1200, classes=classes)

p1.fit(test)
print(p1.weights)
use_classifier(test, p1,  given_point=point, old_entries=True)
plot_cm(test, p1, classes=classes)

    '''
def main():
    menu = ["MinimumDistance4", "MinimumDistance2", "Perceptron", "MaxBayes", "BackPropagation"]
    dict_classes = {
        'versicolor - setosa': {
            'exclude': 'virginica',
            'classes': ['versicolor', 'setosa']
        },
        'versicolor - virginica': {
            'exclude': 'setosa',
            'classes': ['virginica', 'versicolor']
        },

        'virginica - setosa': {
            'exclude': 'versicolor',
            'classes': ['virginica', 'setosa']
        },
        
     
        'versicolor - non_versicolor': {
            'exclude': 'versicolor',
            'classes': ['versicolor', 'non_versicolor']
        },
       

        'setosa - non_setosa': {
            'exclude': 'setosa',
            'classes': ['setosa', 'non_setosa']
        },
        'virginica - non_virginica': {
            'exclude': 'virginica',
            'classes': ['virginica', 'non_virginica' ]
        },
    }
    classes_list = list(dict_classes.keys())
    print(classes_list)
    
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
            selected_option = show_menu_options("Select the classes to be considered", menu=classes_list)
            if selected_option == len(classes_list):
                break
            selected_class = dict_classes[classes_list[selected_option]]
        
            data, test, class1_df, class2_df, opposite_data_df = get_classes(exclude=selected_class['exclude'], classes=selected_class['classes'], overwrite_classes=True)
     
            class1_avg_dict , class1_avg_list = get_averages(class1_df)
            class2_avg_dict , class2_avg_list = get_averages(class2_df)
            print(class1_avg_list)
            print(class2_avg_list)

            while True: 
                title = f'{selected_model} - {selected_class['classes']}'
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
                    use_perceptron(selected_model=selected_model, selected_class=selected_class['classes'], selected_dataset=selected_dataset, exclude=selected_class['exclude'])

                elif any(selected_model in x for x in ["MinimumDistance4", "MinimumDistance2"]):
                    use_minimum_distance_classifier(selected_model=selected_model, selected_dataset=selected_dataset, selected_class=selected_class['classes'], class1_avg_list=class1_avg_list, class2_avg_list=class2_avg_list)
              
                elif selected_model == "MaxBayes":
                    use_bayes(selected_model=selected_model, selected_dataset=selected_dataset, selected_class=selected_class['classes'])
                
                elif selected_model == 'BackPropagation':
                    use_backpropagation(selected_model=selected_model, selected_dataset=selected_dataset, selected_class=selected_class['classes'], exclude=selected_class['exclude'])
                

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

          
def use_minimum_distance_classifier(selected_model, selected_class, selected_dataset, class1_avg_list, class2_avg_list):
    print(len(selected_dataset['dataset']))
  
    if selected_model == 'MinimumDistance4':
        c1 = MinimumDistance4(class1_avg=class1_avg_list, class2_avg=class2_avg_list, classes=selected_class)
    else:
        c1 = MinimumDistance2(class1_avg=class1_avg_list[:2], class2_avg=class2_avg_list[:2], classes=selected_class)

    print(f'class_1: {c1.class1_avg}; class_2: {c1.class2_avg}')
    while True:
        title = f'{selected_model} - {selected_class} - {selected_dataset['title']}'
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

          
def use_perceptron(selected_model, selected_class, selected_dataset, exclude):
    
    p1 = Perceptron(classes=selected_class)
    while True:
        title = f'\n{selected_model} - {selected_class} - {selected_dataset['title']} - {p1.weights}'
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
            data, test, class1_df, class2_df, opposite_data_df = get_classes(exclude=exclude, classes=selected_class, overwrite_classes=True)
            if selected_dataset['title'] == 'training':
                dataset = data
            elif selected_dataset['title'] == 'test':
                dataset = test
            else: 
                dataset = pd.concat([data, test])
            x_test, y_test = extract_values(dataset)
            y_test = map_values(values=y_test, class_mapping=p1.class_mapping)
            p1.fit(inputs=x_test, targets=y_test, epochs=1000, learning_rate=0.01)
        elif selected_option == 2:
            print(f'{title}: predict')
            use_classifier(selected_dataset['dataset'], p1, given_point=point)
        elif selected_option == 3:  
            print(f'{title}: confusion_matrix')                       
            plot_cm(selected_dataset['dataset'], p1)
        else:  
            print(f'{title}: predict')
            print_metrics(selected_dataset['dataset'], p1)
            



def use_bayes(selected_model, selected_class, selected_dataset):
    print(len(selected_dataset['dataset']))
  
    bayes_1 = MaxNaiveBayes(classes=selected_class, df=selected_dataset['dataset'].copy(), Y='Species')

    while True:
        title = f'\n{selected_model} - {selected_class} - {selected_dataset['title']}'
        print(title)
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
                  


def use_backpropagation(selected_model, selected_class, selected_dataset, exclude):
    print(len(selected_dataset['dataset']))
  
    back_prop1 = BackPropagation(df=selected_dataset['dataset'].copy(), class_column='Species')

    while True:
        title = f'\n{selected_model} - {selected_class} - {selected_dataset['title']}'
        print(title)
        # back_prop1.initialise(df=selected_dataset['dataset'].copy(), Y='Species')
        actions_list = ['train', 'classify', 'predict_point', 'confusion_matrix', 'print_metrics']
        selected_option = show_menu_options("Select action:", menu=actions_list)

      
                       
            
        if selected_option == len(actions_list):
            break
        
        if selected_option == 0:
            print(f'{title}: train')
            # use_classifier(selected_dataset['dataset'], back_prop1, decision_boundary=True)
          
            data, test, class1_df, class2_df, opposite_data_df = get_classes(exclude=exclude, classes=selected_class, overwrite_classes=True)
            if selected_dataset['title'] == 'training':
                dataset = data
            elif selected_dataset['title'] == 'test':
                dataset = test
            else: 
                dataset = pd.concat([data, test])
           
            
            x_test, y_test = extract_values(dataset)
            y_test = map_values(values=y_test, class_mapping=back_prop1.class_mapping)
            back_prop1.fit(x_test, y_test, epochs=5000, learning_rate=0.01)
        
        elif selected_option == 1: 
            print(f'{title}: classify')    
                
            '''
            outputs = back_prop1.decision_function(X_test)
            for j, (x, y, output) in enumerate(zip(X_test, Y_test, outputs)):
                print("{} - Input: {} - Target: {} - Prediction: {}".format(j, x, y, output))   
            '''
            
            use_classifier(selected_dataset['dataset'], back_prop1, given_point=point, decision_boundary=True)
            
               
        elif selected_option == 2:                
            print(f'{title}: predict')
            use_classifier(selected_dataset['dataset'], back_prop1, given_point=point, decision_boundary=True)
       
        elif selected_option == 3:                
            print(f'{title}: confusion_matrix')                       
            plot_cm(selected_dataset['dataset'], back_prop1)
        else:  
            print(f'{title}: print_metrics')
            print_metrics(selected_dataset['dataset'], back_prop1)
       
  

def select_classes(classes_list, dict_classes):
    selected_option = show_menu_options("Select the classes to be considered", menu=classes_list)
   
    selected_class = dict_classes[classes_list[selected_option]]
    
    data, test, class1_df, class2_df, opposite_data_df = get_classes(exclude=selected_class['exclude'], classes=selected_class['classes'], overwrite_classes=True)
    
    class1_avg_dict , class1_avg_list = get_averages(class1_df)
    class2_avg_dict , class2_avg_list = get_averages(class2_df)
    print(class1_avg_list)
    print(class2_avg_list)
   

if __name__ == "__main__":
    main()