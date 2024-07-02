
from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QColor, QIcon, QPalette, QIntValidator, QStandardItem, QStandardItemModel

from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QComboBox,
    QErrorMessage,
    QFileDialog,
    QFrame,
    QLabel,
    QListView,
    QFormLayout,
    QWidget,
    QLineEdit,
    QTableWidgetItem,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QToolBar,
)


class SelectableList():
    def __init__(self,  title, items, parent=None):

        self.selectableList = QVBoxLayout(parent)
        self.selectableList.setSpacing(5)
        self.selectableList.addWidget(QLabel(title))

        self.listView = QListView()
        self.selectableList.addWidget(self.listView)
        model = QStandardItemModel(self.listView)
        for item in items:
            # create an item with a caption
            standardItem = QStandardItem(item)
            standardItem.setCheckable(True)
            model.appendRow(standardItem)
        self.listView.setModel(model)
        self.selectableList.addWidget(self.listView)

    def addItems(self, items):
        print(items)
        model = QStandardItemModel(self.listView)
        for item in items:
            # create an item with a caption
            standardItem = QStandardItem(item)
            standardItem.setCheckable(True)
            model.appendRow(standardItem)
        self.listView.setModel(model)

    def itemsSelected(self):
        selected = []
        model = self.listView.model()
        i = 0
        while model.item(i):
            if model.item(i).checkState():
                selected.append(model.item(i).text())
            i += 1
        return selected
