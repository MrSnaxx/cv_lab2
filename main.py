import copy
import math
import warnings

from scipy.signal import convolve2d

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
from scipy.ndimage import median_filter

Ui_MainWindow, _ = uic.loadUiType("interface_lab_2.ui")


import cv2

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
        self.segment_edges_action.clicked.connect(self.segment_edges)
        self.segment_threshold_action.clicked.connect(self.segment_threshold)

    def load_image(self):
        filename = QFileDialog.getOpenFileName(self, "Загрузка изображения", "", "Image (*.png *.tiff *.bmp)")
        if filename[0] == "":
            QMessageBox.about(self, "Ошибка", "Файл не выбран")
            return
        filepath = filename[0]
        self.img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Чтение изображения в оттенках серого
        self.img_original = np.copy(self.img)
        self.set_image()

    def save_image(self):
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Нечего сохранять")
            return
        filename = QFileDialog.getSaveFileName(self, "Open Image", "hue", "Image Files (*.png *.tiff *.bmp)")
        if filename[0] == "":
            QMessageBox.about(self, "Ошибка", "Путь сохранения не выбран")
            return
        cv2.imwrite(filename[0], self.img)

    def set_changes(self):
        self.img = np.copy(self.img_original)
        self.set_image()

    def set_image(self):
        self.image_view.clear()
        self.image_view.setImage(self.img.T)

    def segment_edges(self):
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Сначала загрузите изображение")
            return
        # Сегментация краев с помощью оператора Кэнни
        edges = cv2.Canny(self.img, 100, 200)  # Параметры 100 и 200 - нижний и верхний пороги
        self.img = edges
        self.set_image()

    def segment_threshold(self):
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Сначала загрузите изображение")
            return
        # Сегментация с использованием метода порогового значения (метод Оцу)
        _, thresholded_img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.img = thresholded_img
        self.set_image()


if __name__ == "__main__":
    application = QtWidgets.QApplication(argv)
    program = Redactor()
    program.show()
    exit(application.exec_())
