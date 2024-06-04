import os
import sys
import time
import pandas as pd

from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QColor, QIcon, QPalette, QIntValidator

from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
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
)


from prepareData import load_csv 
from test import use_classifier, plot_cm
from styles import white, primary_colour, secondary_colour 
from models.MinimumDistance4 import MinimumDistance4
from models.MinimumDistance2 import MinimumDistance2
from models.Perceptron import Perceptron

class MainWindow(QMainWindow):
  
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CSV File Loader")
        self.setGeometry(100, 100, 832, 624)
        self.model_selector =  QComboBox()
        self.models = [
            { 'MinimumDistance4': MinimumDistance4},
            { 'MinimumDistance2': MinimumDistance2},
            { 'Perceptron': Perceptron}, 
            
        ]
        self.model_names = [list(d.keys())[0] for d in self.models]

        self.point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}
        self.classes = ['virginica', 'versicolor']



        self.classifiers = []
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.save_path = ""


        self.toolbox()
        self.sidebar()    
         
       
        self.show()
    

    def toolbox(self):
        # Setup tools
        toolbar = QToolBar("Toolbar")
        toolbar.setIconSize(QSize(14, 14))
        self.addToolBar(toolbar)

        
        # Create a table widget
        self.table_widget = QTableWidget(self)
        self.table_widget.setGeometry(200, 50, 600, 400)

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
        change_folder_action.setStatusTip("Change folder where photos are saved.")
        change_folder_action.triggered.connect(self.change_folder)
        toolbar.addAction(change_folder_action)
        
        self.model_selector.addItems(self.model_names)
        self.model_selector.currentIndexChanged.connect(self.update_learning_rate_visibility)

        toolbar.addWidget(self.model_selector)   

    def sidebar(self):        
        self.sidebar = QFrame(self)
        self.sidebar.setGeometry(0, 50, 200, self.height())
        self.sidebar.setStyleSheet(f"background-color: {white};")  # Set background color
        

        # Add buttons to the sidebar
        layout = QVBoxLayout(self.sidebar)
        
        self.add_menu_item(layout, "use_classifier", "code-fork", use_classifier(data2=self.df, classifier= ))
        self.add_menu_item(layout, "plot_cm", "puzzle-piece", plot_cm)
        
        
        self.lbl_integer = QLabel("Integer Validator")
        self.input_learning_rate = QLineEdit(self.sidebar)
        self.input_learning_rate.setPlaceholderText("upto 3 digit value only accept")
        self.input_learning_rate.setValidator(QIntValidator(1, 999, self))
        layout.addWidget(self.lbl_integer)
        layout.addWidget(self.input_learning_rate)
        layout.addStretch()
        self.update_learning_rate_visibility(self.model_selector.currentIndex())

    def add_menu_item(self, layout, label, icon_name, callback):
        button = QPushButton(f"{label}  \u25B6", self.sidebar)  # Unicode right arrow
        button.setIcon(QIcon.fromTheme(icon_name))
        button.setStyleSheet("text-align: left; padding-left: 10px;")
        button.clicked.connect(callback)
        layout.addWidget(button)

    def show_explorer(self):
        self.label.setText("Welcome to Explorer")

    def show_source_control(self):
        self.label.setText("Welcome to Source Control")

    def show_extensions(self):
        self.label.setText("Welcome to Extensions")

    def update_learning_rate_visibility(self, index):
        if index == 1:
            self.lbl_integer.show()
            self.input_learning_rate.show()
        else:
            self.lbl_integer.hide()
            self.input_learning_rate.hide()

    def load_csv(self):
        # Open a file dialog to select a CSV file
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
    
        if file_path:
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Display the data in the table widget
                self.table_widget.setRowCount(len(df) +1)
                self.table_widget.setColumnCount(len(df.columns))            
                self.sidebar.setStyleSheet(f"background-color: {white};")

                for row_idx, row in enumerate(df.columns):
                    print(row_idx, 1, row)                  
                    item = QTableWidgetItem(row)                 
                    self.table_widget.setItem(0, row_idx, item)
                
                for row_idx, row in enumerate(df.values):          
                    for col_idx, value in enumerate(row):
                        item = QTableWidgetItem(str(value))                        
                        self.table_widget.setItem(row_idx + 1, col_idx, item)

                self.table_widget.setAlternatingRowColors(True)
                palette = self.table_widget.palette()
                palette.setColor(QPalette.Base, QColor(primary_colour))
                palette.setColor(QPalette.AlternateBase, QColor(secondary_colour))
                self.table_widget.setPalette(palette)

            except pd.errors.EmptyDataError:
                print("The selected file is empty.")
            except pd.errors.ParserError:
                print("Error parsing the CSV file. Please check the file format.")


    def change_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Snapshot save location", "")
        if path:
            self.save_path = path
            self.save_seq = 0

    def alert(self, s):
        """
        Handle errors coming from QCamera dn QCameraImageCapture by displaying alerts.
        """
        err = QErrorMessage(self)
        err.showMessage(s)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    app.exec()