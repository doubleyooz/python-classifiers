from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,


    QStatusBar,
    QTableWidget,
    QToolBar,
    QWidget
)


class Selector():
    def __init__(self, items, on_change=lambda _: True):
        self.combobox = QComboBox()
        self.model_names = items
        self.on_change = on_change
        self.combobox.addItems(self.model_names)
        self.combobox.currentIndexChanged.connect(
            self.update_selected_model)

    def update_selected_model(self, index):
        print(index)
        self.on_change(index)
        pass
