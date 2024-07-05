
from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QColor, QIcon, QPalette, QIntValidator, QStandardItem, QStandardItemModel

from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QCheckBox,
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
    def __init__(self,  title, items, onChange, parent=None):

        self.selectableList = QVBoxLayout(parent)
        self.selectableList.setSpacing(5)
        self.selectableList.addWidget(QLabel(title))
        self.onChange = onChange
        self.listView = QVBoxLayout()
        self.checkbox_list = []
        self.active_features = []
        for item in items:
            # create an item with a caption
            standardItem = QCheckBox(item)
            standardItem.setCheckable(True)
            standardItem.setChecked(True)
            standardItem.stateChanged.connect(self.itemsSelected)
            self.listView.addWidget(standardItem)
            self.checkbox_list.append(standardItem)

        self.selectableList.addLayout(self.listView)

    def foo(self):
        print('foo')

    def itemsSelected(self):
        selected = []
        i = 0

        temp = list(map(lambda y: y.text(), filter(
            lambda x: not x.isChecked(), self.checkbox_list)))
        # print(temp)
        self.onChange(temp)
        return temp
