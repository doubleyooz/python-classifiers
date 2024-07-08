import os
import sys
import time
import pandas as pd

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPalette, QIntValidator


from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QScrollArea,
    QStatusBar,
    QTableWidget,
    QToolBar,
    QWidget
)
from public.components.SelectableList import SelectableList
from public.styles import white, primary_colour, secondary_colour


class Sidebar():
    def __init__(self, parent):
        sidebarFrame = QFrame(parent)
        sidebarFrame.setGeometry(0, 50, 200, parent.height())
        sidebarFrame.setStyleSheet(
            f"background-color: {white};")

        self.models = []
        self.sidebarFrame = sidebarFrame

        self.mainLayout = QVBoxLayout(sidebarFrame)
        self.selectableLayout = QVBoxLayout()
        self.selectableLists: list[SelectableList] = []
        self.no_csv_warning = QLabel("No dataframe loaded", sidebarFrame)

        '''
        self.scroll_area = QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setLayout(self.selectableLayout)
        '''

        self.mainLayout.addStretch()
        self.add_widget(self.no_csv_warning)

        self.addActionButtons()
        self.addDataArea()
        self.mainLayout.insertLayout(
            self.mainLayout.count()-1, self.selectableLayout)

    def addActionButtons(self):

        # Add buttons to the sidebar
        self.buttonsWidget = QWidget()
        self.buttonsLayout = QVBoxLayout(self.buttonsWidget)
        self.add_menu_item(
            self.buttonsLayout, "use_classifier", lambda _: True)
        self.add_menu_item(self.buttonsLayout,
                           "plot_cm", lambda _: True)

        self.add_widget(self.buttonsWidget)

    def addModelArea(self):

        self.minimunDistanceWidget = QWidget()
        self.lbl_integer = QLabel(
            "Integer Validator", self.minimunDistanceWidget)
        self.input_learning_rate = QLineEdit(parent=self.minimunDistanceWidget)
        self.input_learning_rate.setPlaceholderText(
            "upto 3 digit value only accept")
        self.input_learning_rate.setValidator(QIntValidator(1, 999, self))

        self.add_widget(self.minimunDistanceWidget)

        self.add_widget(self.no_csv_warning)

        self.add_widget(self.class_selector)

    def addDataArea(self):
        self.dataWidget = QWidget()
        self.dataLayout = QVBoxLayout(self.dataWidget)

        self.splitDataLabel = QLabel(
            "What percentage of the data\n should be used for training?")
        self.splitDataInput = QLineEdit()
        self.splitDataInput.setPlaceholderText(
            "Only between 1 and 99")
        self.splitDataInput.setValidator(QIntValidator(1, 10))
        # data, test, class1_df, class2_df, _ = get_classes(df=df,
        #                                                  exclude=selected_class['exclude'], classes=selected_class['classes'], overwrite_classes=True)
        self.dataLayout.addWidget(self.splitDataLabel)
        self.dataLayout.addWidget(self.splitDataInput)
        self.training_checkbox = QCheckBox('use training data')
        self.test_checkbox = QCheckBox('use test data')
        self.dataLayout.addWidget(self.training_checkbox)
        self.dataLayout.addWidget(self.test_checkbox)
        self.add_widget(self.dataWidget)

    def addSelectFeatures(self, title, items, on_change, default_state=True, select_many=True):
        s = SelectableList(
            title=title, on_change=on_change, items=items, default_state=default_state, select_many=select_many)
        self.selectableLists.append(s)
        self.selectableLayout.addWidget(s.mainWidget)

    def update_menu_item(self, idx, label, icon, callback, visible=True):

        self.buttonsLayout.itemAt(idx).widget().setVisible(visible)

        self.buttonsLayout.itemAt(idx).widget().setText(label)
        self.buttonsLayout.itemAt(idx).widget().setIcon(icon)

        self.buttonsLayout.itemAt(idx).widget().setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.buttonsLayout.itemAt(idx).widget().clicked.connect(callback)

    def add_menu_item(self, layout, label, callback):
        # Unicode right arrow
        button = QPushButton(f"{label}  \u25B6")
        button.setStyleSheet("text-align: left; padding-left: 10px;")
        button.clicked.connect(callback)

        layout.addWidget(button)

        return button

    def add_widget(self, widget):
        self.mainLayout.insertWidget(
            self.mainLayout.count()-1, widget)
