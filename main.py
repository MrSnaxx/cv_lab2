import copy
import math
import warnings

# suppress warnings
warnings.filterwarnings('ignore')
from sys import argv

import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from pyqtgraph import SignalProxy
from PyQt5 import uic
# from converted_ui import Ui_MainWindow
import pyqtgraph as pg

Ui_MainWindow, _ = uic.loadUiType("interface_lab_2.ui")

def mean_filter(img,n):
    pass

def median_filter(img,n):
    pass

class Redactor(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Redactor, self).__init__()
        self.setupUi(self)
        self.img = None
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.load_image_action.triggered.connect(self.load_image)
        self.save_image_action.triggered.connect(self.save_image)
        self.smoothing.clicked.connect(self.zalupa)

    def load_image(self):
        filename = QFileDialog.getOpenFileName(self, "Загрузка изображения", "", "Image (*.png *.tiff *.bmp)")
        if filename[0] == "":
            QMessageBox.about(self, "Ошибка", "Файл не выбран")
            return
        filepath = filename[0]
        self.img = Image.open(filepath)
        # Преобразование изображения в массив numpy
        img_array = np.flipud(np.rot90(np.array(self.img)))
        self.img = img_array
        self.img_original = copy.deepcopy(self.img)
        self.img_height = img_array.shape[1]
        self.img_width = img_array.shape[0]
        self.set_image()

    def save_image(self):
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Нечего сохранять")
            return
        filename = QFileDialog.getSaveFileName(self, "Open Image", "hue", "Image Files (*.png *.tiff *.bmp)")
        if filename[0] == "":
            QMessageBox.about(self, "Ошибка", "Путь сохранения не выбран")
            return
        self.image_view.getImageItem().save(filename[0])

    def zalupa(self):
        self.img=self.img_original
        if self.smoothing.isChecked():
            if self.nxn.currentText() == "3x3":
                n = 3
            else:
                n = 5
            if self.filter.currentText() == "Медианный фильтр":
                self.img = median_filter(self.img, n)
            else:
                self.img = mean_filter(self.img, n)
        self.set_image()

    def set_image(self):
        self.image_view.clear()
        self.image_view.setImage(self.img)


if __name__ == "__main__":
    application = QtWidgets.QApplication(argv)
    program = Redactor()
    program.show()
    exit(application.exec_())
