import os
import sys
import time
import pandas as pd


from PySide6.QtGui import QAction, QColor, QIcon, QPalette, QIntValidator

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
from public.components.Sidebar import Sidebar
from public.components.Toolbar import Toolbar
from public.styles import white, primary_colour, secondary_colour
from utils.prepareData import get_pairs, load_csv
from utils.pyside import clear_layout
from utils.test import use_classifier, plot_cm

from constants.main import models


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CSV File Loader")
        self.setGeometry(100, 100, 832, 624)
        self.model_selector = QComboBox()
        self.class_selector = QComboBox()

        self.dict_classes = {}

        self.class_pair_list = []
        self.selected_pair = 0
        self.model_names = list(models.keys())
        self.model_selector.addItems(self.model_names)
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
        self.toolbar.add_toolbox_widget(self.model_selector)
        # Create a table widget
        self.table_widget = QTableWidget(self)
        self.table_widget.setGeometry(200, 50, 600, 400)
        self.sidebar = Sidebar(self)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(self.sidebar.sidebarLayout)
        mainLayout.addWidget(self.table_widget)
        self.setLayout(mainLayout)
        self.on_empty_data()
        self.show()

    def update_selected_class_pair(self, index):

        self.selected_pair = index
        pass

    def update_selected_model(self, index):
        print(index)
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

        for btn_idx, button in enumerate(self.sidebar.buttons):
            self.sidebar.update_menu_item(
                btn_idx, button.text(), button.icon(), dict_funs[btn_idx])

        # self.sidebar.minimunDistanceWidget.hide()
        self.input_learning_rate.hide()
        self.no_csv_warning.show()

    def load_csv(self):
        # Open a file dialog to select a CSV file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)")

        if file_path:
            try:
                # Read the CSV file into a DataFrame
                self.df = pd.read_csv(file_path)
                self.dict_classes = get_pairs(df=self.df)
                self.class_pair_list = list(self.dict_classes.keys())
                print(self.dict_classes)
                print(self.class_pair_list)
                self.class_selector.addItems(self.class_pair_list)
                self.model_selector.currentIndexChanged.connect(
                    self.update_selected_model)

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

                non_num_columns = []
                features = []
                for col in self.df:
                    print(self.df[col].dtype)
                    if self.df[col].dtype == "object":
                        non_num_columns.append(col)
                    else:
                        features.append(col)

                print(non_num_columns)
                self.class_selector.addItems(list(non_num_columns))

                if self.sidebar.features_list != None:
                    clear_layout(self.sidebar.features_list.selectableList)
                self.sidebar.addSelectFeatures(features)

                self.sidebar.sidebarLayout.addWidget(self.class_selector)
                self.on_load_data()
            except pd.errors.EmptyDataError:
                print("The selected file is empty.")
                self.on_empty_data()
            except pd.errors.ParserError:
                print("Error parsing the CSV file. Please check the file format.")
                self.on_empty_data()

    def alert(self, s):
        """
        Handle errors coming from QCamera dn QCameraImageCapture by displaying alerts.
        """
        err = QErrorMessage(self)
        err.showMessage(s)

    def on_empty_data(self):
        print('on_empty_data')
        self.model_selector.setVisible(False)
        self.sidebar.dataWidget.setVisible(False)
        self.sidebar.no_csv_warning.setVisible(True)

    def on_load_data(self):
        print('on_load_data')
        self.model_selector.setVisible(True)
        self.sidebar.dataWidget.setVisible(True)
        self.sidebar.no_csv_warning.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    app.exec()
