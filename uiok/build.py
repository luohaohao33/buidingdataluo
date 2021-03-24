import sys
import first
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = QMainWindow()
    ui = first.Ui_MainWindow1()
    ui.setupUi(mainwindow)
    #ui.retranslateUi(mainwindow)
    mainwindow.show()
    sys.exit(app.exec_())