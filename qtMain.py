import os
import sys
import time
import pandas as pd

from PySide6.QtCore import QSize
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
from public.styles import white, primary_colour, secondary_colour
from utils.prepareData import load_csv
from utils.test import use_classifier, plot_cm

from constants.main import models


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CSV File Loader")
        self.setGeometry(100, 100, 832, 624)
        self.model_selector = QComboBox()
        self.class_selector = QComboBox()

        self.model_names = list(models.keys())

        self.point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}
        self.classes = ['virginica', 'versicolor']

        self.df = None
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.save_path = ""

        self.toolbox()
        # Create a table widget
        self.table_widget = QTableWidget(self)
        self.table_widget.setGeometry(200, 50, 600, 400)
        self.sidebar = Sidebar(self)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(self.sidebar.sidebarLayout)
        mainLayout.addWidget(self.table_widget)
        self.setLayout(mainLayout)
        self.on_halt_data()
        self.show()

    def toolbox(self):
        # Setup tools
        toolbar = QToolBar("Toolbar")
        toolbar.setIconSize(QSize(14, 14))
        self.addToolBar(toolbar)

        load_action = QAction(
            QIcon(os.path.join("images", "camera-black.png")),
            "Load CSV File...",
            self,
        )
        load_action.setStatusTip("Take photo of current view")
        load_action.triggered.connect(self.load_csv)
        toolbar.addAction(load_action)

        change_folder_action = QAction(
            QIcon(os.path.join("images", "blue-folder-horizontal-open.png")),
            "Change save location...",
            self,
        )
        change_folder_action.setStatusTip(
            "Change folder where photos are saved.")
        change_folder_action.triggered.connect(self.change_folder)
        toolbar.addAction(change_folder_action)

        self.model_selector.addItems(self.model_names)
        self.model_selector.currentIndexChanged.connect(
            self.update_selected_model)

        toolbar.addWidget(self.model_selector)

    def show_explorer(self):
        self.label.setText("Welcome to Explorer")

    def show_source_control(self):
        self.label.setText("Welcome to Source Control")

    def show_extensions(self):
        self.label.setText("Welcome to Extensions")

    def update_selected_model(self, index):

        dict_funs = {
            0: (lambda:
                print('')
                )(),
        }

        for btn_idx, button in enumerate(self.sidebar.buttons):
            self.sidebar.update_menu_item(
                btn_idx, button.text(), button.icon(), dict_funs[0])

        self.sidebar.minimunDistanceWidget.hide()
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
                self.sidebar.addSelectFeatures(features)
                self.on_load_data()
            except pd.errors.EmptyDataError:
                print("The selected file is empty.")
                self.on_halt_data()
            except pd.errors.ParserError:
                print("Error parsing the CSV file. Please check the file format.")
                self.on_halt_data()

    def change_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Snapshot save location", "")
        if path:
            self.save_path = path
            self.save_seq = 0

    def alert(self, s):
        """
        Handle errors coming from QCamera dn QCameraImageCapture by displaying alerts.
        """
        err = QErrorMessage(self)
        err.showMessage(s)

    def on_halt_data(self):
        print('on_halt_data')
        self.model_selector.setVisible(False)

        self.sidebar.no_csv_warning.setVisible(True)

    def on_load_data(self):
        print('on_load_data')
        self.model_selector.setVisible(False)
        self.sidebar.no_csv_warning.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    app.exec()
