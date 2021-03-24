from untitled import *
from PyQt5.QtWidgets import QLineEdit, QFileDialog, QPushButton
from PyQt5.QtCore import *
class Ui_MainWindow1(Ui_MainWindow):
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.extra1()

    def extra1(self):
        self.toolButton.clicked.connect(self.showDialog)

    def showDialog(self):
        download_path = QFileDialog.getExistingDirectory(self.toolButton, "浏览", r"C:\Users\15123")
        self.textBrowser.setText(download_path)

