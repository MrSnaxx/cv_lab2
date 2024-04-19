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
from sklearn.cluster import KMeans

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
        self.edges.clicked.connect(self.segment_edges)
        self.ptile.clicked.connect(self.segment_ptile_threshold)
        self.iter.clicked.connect(self.segment_iterative_threshold)
        self.kmeans.clicked.connect(self.segment_kmeans_threshold)
        self.bimbim.clicked.connect(self.adaptive_thresholding)

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

    def preprocess_image(self):
        # Применяем медианный фильтр для сглаживания изображения
        for _ in range(self.num_preps.value()):
            self.img = cv2.medianBlur(self.img, 5)

        # Улучшаем контраст с помощью адаптивной гистограммной эквализации
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # self.img = clahe.apply(self.img)

    def set_changes(self):
        self.img = np.copy(self.img_original)
        self.set_image()

    def set_image(self):
        self.image_view.clear()
        self.image_view.setImage(self.img.T)

    def segment_edges(self):
        self.img = self.img_original
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Сначала загрузите изображение")
            return

        if self.preprocessing.isChecked():
            self.preprocess_image()

        # Сегментация краев с помощью оператора Кэнни
        edges = cv2.Canny(self.img, 100, 200)  # Параметры 100 и 200 - нижний и верхний пороги

        # Заполнение замкнутых краев с помощью операции морфологического закрытия
        # kernel = np.ones((1, 1), np.uint8)
        # closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        self.img = edges
        self.set_image()

    def segment_ptile_threshold(self):
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Сначала загрузите изображение")
            return
        self.img = self.img_original
        if self.preprocessing.isChecked():
            self.preprocess_image()

        # Определение порога с использованием P-tile (например, 90-й перцентиль)
        percentile_value = self.perc.value()
        threshold = np.percentile(self.img, percentile_value)

        # Применение порогового значения
        _, thresholded_img = cv2.threshold(self.img_original, threshold, 255, cv2.THRESH_BINARY)

        self.img = thresholded_img
        self.set_image()

    def segment_iterative_threshold(self):
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Сначала загрузите изображение")
            return
        self.img = self.img_original
        if self.preprocessing.isChecked():
            self.preprocess_image()

        # Применение метода Оцу для определения порога
        _, thresholded_img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.img = thresholded_img
        self.set_image()

    def segment_kmeans_threshold(self):
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Сначала загрузите изображение")
            return
        self.img = self.img_original
        if self.preprocessing.isChecked():
            self.preprocess_image()

        # Параметры для метода k-средних (различные значения k)
        k = self.km.value()

        # Преобразование изображения в одномерный массив
        img_flattened = self.img.flatten().reshape(-1, 1)

        # Применение метода k-средних
        kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flattened)
        aboba = kmeans.labels_.reshape(self.img.shape)
        background_cluster = np.argmax(np.bincount(aboba.ravel()))
        background_values = self.img[aboba == background_cluster]
        threshold = (np.min(background_values) + np.max(background_values)) / 2

        # Применяем пороговое значение
        thresholded_img = np.where(self.img_original > threshold, 255, 0).astype(np.uint8)

        self.img = thresholded_img
        self.set_image()

    def adaptive_thresholding(self):
        k_value = self.k_val.value()
        t_value = self.t_val.value()
        c_method = self.meth.currentText()
        result_image = np.zeros_like(self.img_original)
        for y in range(self.img_original.shape[0]):
            for x in range(self.img_original.shape[1]):
                # Определение границ окна
                y_min = max(0, y - k_value)
                y_max = min(self.img_original.shape[0], y + k_value + 1)
                x_min = max(0, x - k_value)
                x_max = min(self.img_original.shape[1], x + k_value + 1)

                # Выделение окна
                pixel_neighborhood = self.img_original[y_min:y_max, x_min:x_max]

                # Вычисление значения C в зависимости от выбранного метода
                if c_method == 'mean':
                    c_value = np.mean(pixel_neighborhood)
                elif c_method == 'median':
                    c_value = np.median(pixel_neighborhood)
                elif c_method == 'min_max/2':
                    c_value = (np.min(pixel_neighborhood) + np.max(pixel_neighborhood)) / 2

                # Применение порога
                if self.img_original[y, x] - c_value > t_value:
                    result_image[y, x] = 255
                else:
                    result_image[y, x] = 0

        self.img=result_image
        self.set_image()


if __name__ == "__main__":
    application = QtWidgets.QApplication(argv)
    program = Redactor()
    program.show()
    exit(application.exec_())
