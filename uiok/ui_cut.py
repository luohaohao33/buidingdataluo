from PyQt5.QtWidgets import *
from untitled import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.toolButton.clicked.connect(self.showDialog)
        self.toolButton_2.clicked.connect(self.showDialog_2)
        self.textBrowser.setLineWrapMode(QTextEdit.NoWrap)
        self.textBrowser_2.setLineWrapMode(QTextEdit.NoWrap)


    def showDialog(self):
        download_path, _ = QFileDialog.getOpenFileName(self, "浏览", r"C:\Users\15123")
        self.textBrowser.setText(download_path)

    def showDialog_2(self):
        download_path = QFileDialog.getExistingDirectory(self, "浏览", r"C:\Users\15123")
        self.textBrowser_2.setText(download_path)