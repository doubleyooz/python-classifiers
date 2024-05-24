import sys
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Slot

from models.MinimumDistance4 import MinimumDistance4
from models.MinimumDistance2 import MinimumDistance2
from models.Perceptron import Perceptron
from prepareData import get_pairs, setosa_avg, versicolor_avg, virginica_avg
from test import use_classifier, plot_cm


class MyPythonObject(QObject):
    @Slot()
    def my_python_object(self):
        point = {'x1': 5.7, 'x2': 4.4, 'x3': 3.5, 'x4': 1.5}
        data, test = get_pairs(exclude='setosa', random=False)
        pairs = ['virginica', 'versicolor']
        p1 = Perceptron(learning_rate=0.01, max_iters=1200, pairs=pairs)
        p1.fit(test)
        print(p1.weights)
        # Call other functions or perform additional actions here

# Create an instance of your Python object
my_python_object = MyPythonObject()

QML = """
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Window {
    width: 800
    height: 800
    visible: true
    title: "Hello World"

    readonly property list<string> texts: ["Hallo Welt", "Hei maailma",
                                           "Hola Mundo", "Привет мир"]

    function setText() {
        var i = Math.round(Math.random() * 3)
        text.text = texts[i]
    }

    ColumnLayout {
        anchors.fill:  parent

        Text {
            id: text
            text: "Hello World"
            Layout.alignment: Qt.AlignHCenter
        }
        Button {
            text: "Click me"
            Layout.alignment: Qt.AlignHCenter
            onClicked:  myPythonObject.my_python_object()
        }
    }
}
"""

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Register the Python object
    engine.rootContext().setContextProperty("myPythonObject", my_python_object)

    engine.loadData(QML.encode('utf-8'))
    if not engine.rootObjects():
        sys.exit(-1)
    exit_code = app.exec()
    del engine
    sys.exit(exit_code)