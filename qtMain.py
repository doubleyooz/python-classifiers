import os
import sys
import time
import pandas as pd


from PySide6.QtGui import QAction, QColor, QIcon, QPalette, QIntValidator, QStandardItem, QStandardItemModel

from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QErrorMessage,
    QFileDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QTableWidgetItem,
    QMainWindow,
    QPushButton,

    QStatusBar,
    QTableWidget,
    QToolBar,
    QWidget
)


from public.components.SelectableList import SelectableList
from public.components.Selector import Selector
from public.components.Sidebar import Sidebar
from public.components.Toolbar import Toolbar
from public.styles import white, primary_colour, secondary_colour
from utils.prepareData import get_classes, get_pairs, load_csv
from utils.pyside import clear_layout
from utils.test import use_classifier, plot_cm

from constants.main import models


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CSV File Loader")
        self.setGeometry(100, 100, 832, 624)
        self.model_selector = Selector(
            list(models.keys()), self.update_combobox_model)
        self.class_selector = Selector([], self.update_combobox_class)

        self.dict_classes = {}

        self.class_pair_list = []
        self.training = []
        self.test = []
        self.selected_pair = 0

        self.point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}
        self.classes = ['virginica', 'versicolor']

        self.df = None
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.save_path = ""

        self.toolbar = Toolbar(self)
        self.toolbar.add_toolbox_item(QAction(
            QIcon(os.path.join("images", "camera-black.png")),
            "Load CSV File...",
            self,
        ), self.load_csv)
        self.toolbar.add_toolbox_widget(self.model_selector.combobox)
        # Create a table widget
        self.table_widget = QTableWidget(self)
        self.table_widget.setGeometry(200, 50, 600, 400)
        self.sidebar = Sidebar(self)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(self.sidebar.mainLayout)
        mainLayout.addWidget(self.table_widget)
        self.setLayout(mainLayout)
        self.on_empty_data()
        self.show()

    def update_combobox_class(self, index):

        print(
            f"text: {self.class_selector.combobox.currentText()}, index: {index}")
        current_pair = self.dict_classes[self.class_selector.combobox.currentText(
        )]
        data, test, _, _, _ = get_classes(df=self.df,
                                          exclude=current_pair['exclude'], classes=current_pair['classes'], overwrite_classes=True)

        self.training = data
        self.test = test

    def update_combobox_model(self, index):

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
        print(f'index: {index}')
        print(dict_funs[index])

        '''
           for btn_idx, button in enumerate(self.sidebar.buttons):
            self.sidebar.update_menu_item(
                btn_idx, button.text(), button.icon(), dict_funs[btn_idx])

        # self.sidebar.minimunDistanceWidget.hide()
        self.input_learning_rate.hide()
        self.no_csv_warning.show()
        '''

    def update_split_data(self):

        pass

    def load_csv(self):
        # Open a file dialog to select a CSV file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)")

        if file_path:
            try:
                # Read the CSV file into a DataFrame
                self.df = pd.read_csv(file_path)

                print('got here')

                # Display the data in the table widget
                self.table_widget.setRowCount(len(self.df) + 1)
                self.table_widget.setColumnCount(len(self.df.columns))

                for row_idx, row in enumerate(self.df.columns):
                    print(row_idx, 1, row)
                    item = QTableWidgetItem(row)
                    self.table_widget.setItem(0, row_idx, item)

                for row_idx, row in enumerate(self.df.values):
                    for col_idx, value in enumerate(row):
                        item = QTableWidgetItem(str(value))
                        self.table_widget.setItem(row_idx + 1, col_idx, item)

                self.table_widget.setAlternatingRowColors(True)
                palette = self.table_widget.palette()
                palette.setColor(QPalette.Base, QColor(primary_colour))
                palette.setColor(QPalette.AlternateBase,
                                 QColor(secondary_colour))
                self.table_widget.setPalette(palette)

                self.not_features = []
                self.features = []
                for col in self.df:
                    print(self.df[col].dtype)
                    if self.df[col].dtype == "object":
                        self.not_features.append(col)
                    else:
                        self.features.append(col)

                print(self.not_features)

                if self.sidebar.selectableLayout != None:
                    clear_layout(
                        self.sidebar.selectableLayout)

                self.sidebar.addSelectFeatures(
                    "Select Features", self.features, on_change=self.update_selected_feature)
                self.sidebar.add_widget(self.class_selector.combobox)
                self.sidebar.addSelectFeatures(
                    "Select Class Column", self.not_features, on_change=self.update_selected_class, default_state=False, select_many=False)
                print(f'not_features: {self.not_features}')

                self.on_load_data()
            except pd.errors.EmptyDataError:
                print("The selected file is empty.")
                self.on_empty_data()
            except pd.errors.ParserError:
                print("Error parsing the CSV file. Please check the file format.")
                self.on_empty_data()

    def update_selected_feature(self, items: list[str]):
        # self.table_widget.setColumnHidden(1, True)

        column_count = self.table_widget.columnCount()
        for n in range(0, column_count):
            column_name = self.table_widget.item(0, n).text()
            if column_name in items:
                self.table_widget.setColumnHidden(n, True)
            elif column_name in self.features:
                self.table_widget.setColumnHidden(n, False)
                print(column_name)

        # print(items)

    def update_selected_class(self, items: list[str]):
        # self.table_widget.setColumnHidden(1, True)
        temp = []
        column_count = self.table_widget.columnCount()
        for n in range(0, column_count):
            column_name = self.table_widget.item(0, n).text()
            if column_name in items:
                self.table_widget.setColumnHidden(n, True)
            elif column_name in self.not_features:
                self.table_widget.setColumnHidden(n, False)
                temp.append(column_name)
        if len(temp) > 0:
            self.switch_class_column(temp[0])

    def switch_class_column(self, class_column):

        self.dict_classes = get_pairs(
            df=self.df, class_column=class_column)
        self.class_pair_list = list(self.dict_classes.keys())
        print(self.dict_classes)
        print(self.class_pair_list)
        self.class_selector.combobox.addItems(self.class_pair_list)

    def on_empty_data(self):
        print('on_empty_data')
        self.model_selector.combobox.setVisible(False)
        self.class_selector.combobox.setVisible(False)
        self.sidebar.dataWidget.setVisible(False)
        self.sidebar.buttonsWidget.setVisible(False)
        self.sidebar.no_csv_warning.setVisible(True)

    def on_load_data(self):
        print('on_load_data')
        self.class_selector.combobox.setVisible(True)
        self.update_selected_class(self.not_features[0])

        self.model_selector.combobox.setVisible(True)
        self.sidebar.dataWidget.setVisible(True)
        self.sidebar.buttonsWidget.setVisible(True)
        self.sidebar.no_csv_warning.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    app.exec()
