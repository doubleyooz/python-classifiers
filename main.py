import numpy as np
import pandas as pd

from constants.main import models
from utils.prepareData import get_classes, get_averages, get_pairs, load_csv
from utils.test import use_classifier, plot_cm, calculate_metrics, print_metrics, extract_values, map_values

point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}
# Set the decimal separator to a comma

df = load_csv('assets/Iris.csv')


def main():

    menu = list(models.keys())

    dict_classes = get_pairs(df=df)

    classes_list = list(dict_classes.keys())

    while True:
        selected_option = show_menu_options(
            "Select the model to be used", menu=menu, last_option="Shut down")
        print(selected_option)
        if selected_option == len(menu):
            print('end of operation')
            break
        selected_model = menu[selected_option]

        while True:
            # Use MinimumDistance4 model
            print(selected_model)
            selected_option = show_menu_options(
                "Select the classes to be considered", menu=classes_list)
            if selected_option == len(classes_list):
                break
            selected_class = dict_classes[classes_list[selected_option]]

            data, test, class1_df, class2_df, _ = get_classes(df=df,
                                                              exclude=selected_class['exclude'], classes=selected_class['classes'], overwrite_classes=True)
            get_pairs(df=df)
            class1_avg_dict, class1_avg_list = get_averages(class1_df)
            class2_avg_dict, class2_avg_list = get_averages(class2_df)
            print(class1_avg_list)
            print(class2_avg_list)

            while True:
                title = f"{selected_model} - {selected_class['classes']}"
                print(title)
                datasets_list = ['training', 'test', 'full']
                selected_option = show_menu_options(
                    "Select dataset to be used", menu=datasets_list)
                if selected_option == len(datasets_list):
                    break

                if selected_option == 0:
                    selected_dataset = {'dataset': data,
                                        'title': datasets_list[selected_option]}
                elif selected_option == 1:
                    selected_dataset = {'dataset': test,
                                        'title': datasets_list[selected_option]}
                else:
                    selected_dataset = {'dataset': pd.concat(
                        [data, test]), 'title': datasets_list[selected_option]}

                print(selected_model)
                if selected_model == "Perceptron":
                    use_perceptron(selected_model=selected_model, selected_class=selected_class[
                                   'classes'], selected_dataset=selected_dataset, exclude=selected_class['exclude'])

                elif any(selected_model in x for x in ["MinimumDistance4", "MinimumDistance2", "MinimumDistanceN"]):
                    use_minimum_distance_classifier(selected_model=selected_model, selected_dataset=selected_dataset, selected_class=selected_class[
                                                    'classes'], class1_avg_list=class1_avg_list, class2_avg_list=class2_avg_list, exclude=selected_class['exclude'])

                elif selected_model == "MaxBayes":
                    use_bayes(selected_model=selected_model, selected_dataset=selected_dataset,
                              selected_class=selected_class['classes'], exclude=selected_class['exclude'])

                elif selected_model == "KMeans":
                    use_k_means(selected_model=selected_model, selected_dataset=selected_dataset,
                                selected_class=selected_class['classes'], exclude=selected_class['exclude'])

                elif selected_model == 'BackPropagation':
                    use_backpropagation(selected_model=selected_model, selected_dataset=selected_dataset,
                                        selected_class=selected_class['classes'], exclude=selected_class['exclude'])


def show_menu_options(title, menu, last_option="Go back"):
    print(f"\n{title}")
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


def use_minimum_distance_classifier(selected_model, selected_class, selected_dataset, class1_avg_list, class2_avg_list, exclude):
    print(len(selected_dataset['dataset']))

    if selected_model == 'MinimumDistance4':
        c1 = models[selected_model](class1_avg=class1_avg_list,
                                    class2_avg=class2_avg_list, classes=selected_class)
    elif selected_model == 'MinimumDistance2':
        c1 = models[selected_model](
            class1_avg=class1_avg_list[:2], class2_avg=class2_avg_list[:2], classes=selected_class)
    else:
        c1 = models[selected_model](df=selected_dataset['dataset'])

    while True:
        title = f"{
            selected_model} - {selected_class} - {selected_dataset['title']}"
        print(title)
        actions_list = ['classify', 'predict_point',
                        'confusion_matrix', 'print_metrics']
        selected_option = show_menu_options(
            "Select action:", menu=actions_list)
        if selected_option == len(actions_list):
            break

        if selected_option == 0:
            print(f"{title}: classify")
            use_classifier(selected_dataset['dataset'], c1)
        elif selected_option == 1:
            print(f"{title}: predict")
            use_classifier(selected_dataset['dataset'], c1, given_point=point)
        elif selected_option == 2:
            plot_confusion_matrix(title, exclude, selected_class, c1)

        else:
            print(f"{title}: print_metrics")
            calculate_metrics(selected_dataset['dataset'], c1)
            print_metrics(c1)


def use_k_means(selected_model, selected_class, selected_dataset, exclude):
    print(len(selected_dataset['dataset']))

    k1 = models[selected_model](
        k=list(selected_dataset['dataset'].columns[: -1]))
    print(k1.centroids, k1.k)

    while True:
        title = f"{
            selected_model} - {selected_class} - {selected_dataset['title']}"
        print(title)
        actions_list = ['classify', 'predict_point',
                        'confusion_matrix', 'print_metrics']
        selected_option = show_menu_options(
            "Select action:", menu=actions_list)
        if selected_option == len(actions_list):
            break

        if selected_option == 0:
            print(f"{title}: classify")
            use_classifier(selected_dataset['dataset'], k1)
        elif selected_option == 1:
            print(f"{title}: predict")
            use_classifier(selected_dataset['dataset'], k1, given_point=point)
        elif selected_option == 2:
            plot_confusion_matrix(title, exclude, selected_class, k1)

        else:
            print(f"{title}: print_metrics")
            calculate_metrics(selected_dataset['dataset'], k1)
            print_metrics(k1)


def use_perceptron(selected_model, selected_class, selected_dataset, exclude):

    p1 = models[selected_model](
        df=selected_dataset['dataset'].copy(), class_column='Species')
    while True:
        title = f"\n{
            selected_model} - {selected_class} - {selected_dataset['title']} - {p1.weights}"
        print(title)
        actions_list = ['classify', 'fit', 'predict_point',
                        'confusion_matrix', 'print_metrics']
        selected_option = show_menu_options(
            "Select action:", menu=actions_list)
        if selected_option == len(actions_list):
            break

        if selected_option == 0:
            print(f"{title}: classify")
            while True:
                title = f"\nDo you want to train before classifying?"
                print(title)
                actions_list = ['yes', 'no']
                selected_option = show_menu_options(
                    "Select action:", menu=actions_list)
                fit = True
                if selected_option == len(actions_list):
                    break

                if selected_option == 0:
                    print(f"{title}: yes")

                else:
                    print(f"{title}: no")
                    fit = False
                use_classifier(selected_dataset['dataset'], p1, fit=fit)
        elif selected_option == 1:
            print(f"{title}: fit")
            data, test, class1_df, class2_df, opposite_data_df = get_classes(df=df,
                                                                             exclude=exclude, classes=selected_class, overwrite_classes=True)
            if selected_dataset['title'] == 'training':
                dataset = data
            elif selected_dataset['title'] == 'test':
                dataset = test
            else:
                dataset = pd.concat([data, test])
            x_test, y_test = extract_values(dataset)
            y_test = map_values(values=y_test, class_mapping=p1.class_mapping)

            while True:
                try:
                    epochs = int(input("Enter the number of epochs: ") or 1000)
                    hanstRanYet = True
                    while hanstRanYet:
                        try:
                            lr = float(
                                input("Enter the learning rate (e.g., 0.01): ") or 0.01)
                            print(f"{lr}: {type(lr)}")
                            # Submit the fit method to a thread pool
                            print(f'epochs: {epochs}, lr: {lr}')
                            p1.fit(inputs=x_test, targets=y_test,
                                   epochs=epochs, learning_rate=lr, verbose=True)
                            break

                        except Exception as e:
                            print(e)
                            print(
                                "Please enter a valid float for the learning_rate.")
                    break

                except ValueError:
                    print("Please enter a valid integer for the number of epochs.")

        elif selected_option == 2:
            print(f"{title}: predict")
            use_classifier(selected_dataset['dataset'], p1, given_point=point)
        elif selected_option == 3:
            plot_confusion_matrix(title, exclude, selected_class, p1)

        else:
            print(f"{title}: print_metrics")
            calculate_metrics(selected_dataset['dataset'], p1)
            print_metrics(p1)


def use_bayes(selected_model, selected_class, selected_dataset, exclude):
    print(len(selected_dataset['dataset']))

    while True:
        title = f"\n{
            selected_model} - {selected_class} - {selected_dataset['title']}"
        print(title)
        actions_list = ['gaussian', 'categorical']
        selected_option = show_menu_options(
            "Select action:", menu=actions_list)
        gaussian = True
        if selected_option == len(actions_list):
            break

        if selected_option == 0:
            print(f"{title}: gaussian")

        else:
            print(f"{title}: categorical")
            gaussian = False

        bayes_1 = models[selected_model](df=selected_dataset['dataset'].copy(
        ), class_column='Species', gaussian=gaussian)
        while True:
            title = f"\n{
                selected_model} - {selected_class} - {selected_dataset['title']}"
            print(title)
            actions_list = ['classify', 'predict_point',
                            'confusion_matrix', 'print_metrics']
            selected_option = show_menu_options(
                "Select action:", menu=actions_list)
            if selected_option == len(actions_list):
                break

            if selected_option == 0:
                print(f"{title}: classify")
                use_classifier(
                    selected_dataset['dataset'], bayes_1, decision_boundary=True)

            elif selected_option == 1:
                print(f"{title}: predict")
                use_classifier(
                    selected_dataset['dataset'], bayes_1, given_point=point, decision_boundary=True)
            elif selected_option == 2:
                plot_confusion_matrix(title, exclude, selected_class, bayes_1)

            else:
                print(f"{title}: print_metrics")
                calculate_metrics(selected_dataset['dataset'], bayes_1)
                print_metrics(bayes_1)


def use_backpropagation(selected_model, selected_class, selected_dataset, exclude):
    print(len(selected_dataset['dataset']))

    back_prop1 = models[selected_model](
        df=selected_dataset['dataset'].copy(), class_column='Species')

    while True:
        title = f"\n{
            selected_model} - {selected_class} - {selected_dataset['title']}"
        print(title)
        # back_prop1.initialise(df=selected_dataset['dataset'].copy(), Y='Species')
        actions_list = ['train', 'classify', 'predict_point',
                        'confusion_matrix', 'print_metrics']
        selected_option = show_menu_options(
            "Select action:", menu=actions_list)

        if selected_option == len(actions_list):
            break

        if selected_option == 0:
            print(f"{title}: train")
            # use_classifier(selected_dataset['dataset'], back_prop1, decision_boundary=True)

            data, test, class1_df, class2_df, opposite_data_df = get_classes(df=df,
                                                                             exclude=exclude, classes=selected_class, overwrite_classes=True)
            if selected_dataset['title'] == 'training':
                dataset = data
            elif selected_dataset['title'] == 'test':
                dataset = test
            else:
                dataset = pd.concat([data, test])

            x_test, y_test = extract_values(dataset)
            y_test = map_values(
                values=y_test, class_mapping=back_prop1.class_mapping)

            while True:
                try:
                    epochs = int(input("Enter the number of epochs: "))
                    hanstRanYet = True
                    while hanstRanYet:
                        try:
                            lr = float(
                                input("Enter the learning rate (e.g., 0.01): "))
                            print(f"{lr}: {type(lr)}")
                            # Submit the fit method to a thread pool
                            back_prop1.fit(
                                inputs=x_test, targets=y_test, epochs=epochs, learning_rate=lr)
                            break

                        except Exception as e:
                            print(e)
                            print(
                                "Please enter a valid float for the learning_rate.")
                    break

                except ValueError:
                    print("Please enter a valid integer for the number of epochs.")

        elif selected_option == 1:
            print(f"{title}: classify")

            '''
            outputs = back_prop1.decision_function(X_test)
            for j, (x, y, output) in enumerate(zip(X_test, Y_test, outputs)):
                print("{} - Input: {} - Target: {} - Prediction: {}".format(j, x, y, output))   
            '''

            while True:
                title = f"\nDo you want to train before classifying?"
                print(title)
                actions_list = ['yes', 'no']
                selected_option = show_menu_options(
                    "Select action:", menu=actions_list)
                fit = True
                if selected_option == len(actions_list):
                    break

                if selected_option == 0:
                    print(f"{title}: yes")

                else:
                    print(f"{title}: no")
                    fit = False

                use_classifier(selected_dataset['dataset'], back_prop1,
                               given_point=point, decision_boundary=True, fit=fit)

            print('weights')
            for index, layer in enumerate(back_prop1.weights):
                print(f'layer({index}) ({np.array(layer).shape}): ', layer)
        elif selected_option == 2:
            print(f"{title}: predict")
            use_classifier(
                selected_dataset['dataset'], back_prop1, given_point=point, decision_boundary=True)

        elif selected_option == 3:
            plot_confusion_matrix(title, exclude, selected_class, back_prop1)

        else:
            print(f"{title}: print_metrics")
            calculate_metrics(selected_dataset['dataset'], back_prop1)
            print_metrics(back_prop1)


def select_classes(classes_list, dict_classes):
    selected_option = show_menu_options(
        "Select the classes to be considered", menu=classes_list)

    selected_class = dict_classes[classes_list[selected_option]]

    data, test, class1_df, class2_df, opposite_data_df = get_classes(df=df,
                                                                     exclude=selected_class['exclude'], classes=selected_class['classes'], overwrite_classes=True)

    class1_avg_dict, class1_avg_list = get_averages(class1_df)
    class2_avg_dict, class2_avg_list = get_averages(class2_df)
    print(class1_avg_list)
    print(class2_avg_list)


def plot_confusion_matrix(title, exclude, selected_class, model):
    while (True):
        print(f"{title}: confusion_matrix")
        datasets_list = ['training', 'test', 'full']

        data, test, class1_df, class2_df, opposite_data_df = get_classes(df=df,
                                                                         exclude=exclude, classes=selected_class, overwrite_classes=True)
        selected_option = show_menu_options(
            "Select dataset to be used in the confusion matrix", menu=datasets_list)
        if selected_option == len(datasets_list):
            break

        if selected_option == 0:
            selected_dataset = {'dataset': data,
                                'title': datasets_list[selected_option]}
        elif selected_option == 1:
            selected_dataset = {'dataset': test,
                                'title': datasets_list[selected_option]}
        else:
            selected_dataset = {'dataset': pd.concat(
                [data, test]), 'title': datasets_list[selected_option]}
        plot_cm(selected_dataset['dataset'], model)


if __name__ == "__main__":
    main()
