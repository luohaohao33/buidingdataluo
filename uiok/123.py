import sys
from PyQt5.QtWidgets import QWidget,QMainWindow, QApplication, QLineEdit, QFileDialog,QPushButton
class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.lineEdit = QLineEdit("",self)
        self.lineEdit.move(30,30)
        self.lineEdit.resize(200,20)
        self.btn = QPushButton("浏览",self)
        self.btn.move(240,30)
        self.btn.resize(50, 20)
        self.btn.clicked.connect(self.showDialog)
        self.setGeometry(300, 300, 300, 100)
        self.setWindowTitle('浏览')
        self.show()
    def showDialog(self):
        download_path = QFileDialog.getExistingDirectory(self,"浏览",r"C:/Users/Administrator/Desktop")
        self.lineEdit.setText(download_path)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())