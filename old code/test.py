from PyQt5.QtWidgets import QApplication, QWidget

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()  # Initialize QWidget properly
        self.init_ui()

    def init_ui(self):
        # Customize the widget, add buttons, labels, etc.
        self.setWindowTitle('My Custom Widget')
        self.resize(300, 200)

app = QApplication([])
widget = MyWidget()
widget.show()
app.exec_()