
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
    QScrollArea,
    QTableWidget,
    QToolBar,
)


class SelectableList():
    def __init__(self,  title, items, on_change, default_state=True, select_many=True, parent=None):
        self.mainWidget = QWidget(parent)
        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.addWidget(QLabel(title))
        self.on_change = on_change
        self.select_many = select_many
        self.listView = QVBoxLayout()

        self.checkbox_list = []
        for idx, item in enumerate(items):
            # create an item with a caption
            standardItem = QCheckBox(item)
            standardItem.setCheckable(True)

            standardItem.setChecked(default_state)
            standardItem.stateChanged.connect(
                lambda state, index=idx, text=item: self.get_selected_items(index, text))
            self.listView.addWidget(standardItem)
            self.checkbox_list.append(standardItem)

        self.mainLayout.addLayout(self.listView)

    def get_selected_items(self, index, text):
        print(f'\nselect_many: {self.select_many}, index: {
              index}, text: {text}')
        if not self.select_many:
            for checkbox in self.checkbox_list:
                if checkbox.text() == text:
                    print(f'checked: ${checkbox.text()}')
                    if not checkbox.isChecked():
                        checkbox.setChecked(True)
                else:
                    if checkbox.isChecked():
                        checkbox.setChecked(False)
        items = list(map(lambda y: y.text(), filter(
            lambda x: not x.isChecked(), self.checkbox_list)))

        self.on_change(items)

        return items
