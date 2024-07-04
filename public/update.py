from public.model import selected_model


def update_selected_class_pair(index):

    selected_pair = index
    pass


def update_selected_model(text):
    selected_model = text
    dict_funs = {
        0: (lambda:
            print('0'))(),
        1: (lambda:
            print('1'))(),
        2: (lambda:
            print('2'))(),
        3: (lambda:
            print('3'))(),
        4: (lambda:
            print('4'))(),
        5: (lambda:
            print('5'))(),
        6: (lambda:
            print('6'))(),
        7: (lambda:
            print('7'))(),
        8: (lambda:
            print('8'))(),

    }
    for btn_idx, button in enumerate(sidebar.buttons):
        sidebar.update_menu_item(
            btn_idx, button.text(), button.icon(), dict_funs[btn_idx])

    sidebar.minimunDistanceWidget.hide()
    input_learning_rate.hide()
    no_csv_warning.show()
