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
from public.styles import white, primary_colour, secondary_colour


class Sidebar():
    def __init__(self, parent):
        sidebarFrame = QFrame(parent)
        sidebarFrame.setGeometry(0, 50, 200, parent.height())
        sidebarFrame.setStyleSheet(
            f"background-color: {white};")

        self.sidebarFrame = sidebarFrame
        self.sidebarLayout = QVBoxLayout(sidebarFrame)
        self.buttons: list[QPushButton] = []
        self.models = []
        self.no_csv_warning = QLabel("No dataframe loaded", sidebarFrame)
        self.addActionButtons()
        self.addDataArea()

    def addActionButtons(self):

        # Add buttons to the sidebar

        self.buttons.append(self.add_menu_item(
            self.sidebarLayout, "use_classifier", "code-fork", lambda _: True))
        self.buttons.append(self.add_menu_item(self.sidebarLayout,
                                               "plot_cm", "puzzle-piece", lambda _: True))

    def addModelArea(self):

        self.minimunDistanceWidget = QWidget()
        self.lbl_integer = QLabel(
            "Integer Validator", self.minimunDistanceWidget)
        self.input_learning_rate = QLineEdit(parent=self.minimunDistanceWidget)
        self.input_learning_rate.setPlaceholderText(
            "upto 3 digit value only accept")
        self.input_learning_rate.setValidator(QIntValidator(1, 999, self))

        self.sidebarLayout.addWidget(self.minimunDistanceWidget)
        self.sidebarLayout.addWidget(self.no_csv_warning)

        self.sidebarLayout.addWidget(self.class_selector)

    def addDataArea(self):
        self.dataWidget = QWidget()
        self.dataLayout = QVBoxLayout(self.dataWidget)

        self.splitDataLabel = QLabel(
            "Split Training and Test Data")
        self.splitDataInput = QLineEdit()
        self.splitDataInput.setPlaceholderText(
            "Only between 1 and 10")
        self.splitDataInput.setValidator(QIntValidator(1, 10))

        self.dataLayout.addWidget(self.splitDataLabel)
        self.dataLayout.addWidget(self.splitDataInput)
        self.sidebarLayout.addWidget(self.dataWidget)

    def addSelectFeatures(self, items):
        self.features_list = SelectableList(
            title='Select Features', items=items)
        self.sidebarLayout.addLayout(self.features_list.selectableList)

    def update_menu_item(self, idx, label, icon, callback):
        self.buttons[idx].setText(label)
        self.buttons[idx].setIcon(icon)

        self.buttons[idx].setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.buttons[idx].clicked.connect(callback)

    def add_menu_item(self, layout, label, icon_name, callback):
        # Unicode right arrow
        button = QPushButton(f"{label}  \u25B6")
        button.setIcon(QIcon.fromTheme(icon_name))
        button.setStyleSheet("text-align: left; padding-left: 10px;")
        button.clicked.connect(callback)

        layout.addWidget(button)
        return button
