import os
import sys
import time

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


class Toolbar:
    def __init__(self, parent):
        self.myToolbar = QToolBar("Toolbar", parent)
        self.myToolbar.setIconSize(QSize(14, 14))

        parent.addToolBar(self.myToolbar)

    def add_toolbox_item(self, q_action: QAction, onClick, status_tip=''):
        # Setup tools

        q_action.setStatusTip(status_tip)
        q_action.triggered.connect(onClick)
        self.myToolbar.addAction(q_action)

    def add_toolbox_widget(self, widget):
        self.myToolbar.addWidget(widget)
