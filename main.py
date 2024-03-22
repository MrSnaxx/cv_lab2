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
from scipy.ndimage import median_filter

Ui_MainWindow, _ = uic.loadUiType("interface_lab_2.ui")


def sigma_filter(image, sigma):
    """
    Applies a sigma filter to an image.

    Args:
        image: np.array representing the image.
        sigma: The sigma value for the filter.

    Returns:
        np.array, the filtered image.
    """

    kernel_size = int(2 * sigma + 1)  # Ensure odd kernel size
    x, y = np.mgrid[-kernel_size // 2 + 1: kernel_size // 2 + 1, -kernel_size // 2 + 1: kernel_size // 2 + 1]

    # Sigma filter kernel (unnormalized)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # Вычисление размеров изображения
    height, width, channels = image.shape
    # Calculate padding
    padding = kernel_size // 2

    # Pad the image
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Create output image
    filtered_image = np.zeros_like(image)

    # Свертка изображения с фильтром для каждого канала
    for channel in range(channels):
        for i in range(height):
            for j in range(width):
                filtered_image[i, j, channel] = np.sum(
                    padded_image[i:i + kernel_size, j:j + kernel_size, channel] * kernel)

    return filtered_image

def gaussian_filter(image, sigma):
    """
    Сглаживает RGBA-изображение с помощью фильтра Гаусса.

    Аргументы:
      image: np.array, представляющий RGBA-изображение.
      sigma: стандартное отклонение (сигма) для фильтра Гаусса.

    Возвращает:
      np.array, сглаженное RGBA-изображение.
    """

    # Вычисление размера ядра фильтра по правилу 3*sigma
    kernel_size = int(6 * sigma + 1)  # Обеспечиваем нечетный размер ядра

    # Создание сетки координат для ядра
    x, y = np.mgrid[-kernel_size // 2 + 1: kernel_size // 2 + 1, -kernel_size // 2 + 1: kernel_size // 2 + 1]

    # Вычисление значений фильтра Гаусса
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # Нормализация ядра

    # Вычисление размеров изображения
    height, width, channels = image.shape

    # Вычисление отступов для свертки
    padding = kernel_size // 2

    # Добавление отступов к изображению
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Создание выходного изображения
    smoothed_image = np.zeros_like(image)

    # Свертка изображения с фильтром для каждого канала
    for channel in range(channels):
        for i in range(height):
            for j in range(width):
                smoothed_image[i, j, channel] = np.sum(
                    padded_image[i:i + kernel_size, j:j + kernel_size, channel] * kernel)

    return smoothed_image


def mean_filter(image, n):
    """
    Сглаживает RGBA-изображение с помощью прямоугольного фильтра.

    Аргументы:
    image: np.array, представляющий RGBA-изображение.
    n: размер фильтра (n x n).

    Возвращает:
    np.array, сглаженное RGBA-изображение.
    """

    # Проверка входных данных
    if n <= 0 or n % 2 == 0:
        raise ValueError("n должно быть нечетным положительным числом.")

    # Вычисление размеров изображения
    height, width, channels = image.shape

    # Создание прямоугольного фильтра
    filter = np.ones((n, n)) / (n * n)

    # Вычисление отступов для свертки
    padding = n // 2

    # Добавление отступов к изображению
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Создание выходного изображения
    smoothed_image = np.zeros_like(image)

    # Свертка изображения с фильтром для каждого канала
    for channel in range(channels):
        for i in range(height):
            for j in range(width):
                smoothed_image[i, j, channel] = np.sum(padded_image[i:i + n, j:j + n, channel] * filter)

    return smoothed_image


def median_filter(image, n):
    """
    Сглаживает RGBA-изображение с помощью медианного фильтра.

    Аргументы:
      image: np.array, представляющий RGBA-изображение.
      n: размер фильтра (n x n).

    Возвращает:
      np.array, сглаженное RGBA-изображение.
    """

    # Проверка входных данных
    if n <= 0 or n % 2 == 0:
        raise ValueError("n должно быть нечетным положительным числом.")

    # Вычисление размеров изображения
    height, width, channels = image.shape

    # Вычисление отступов для свертки
    padding = n // 2

    # Добавление отступов к изображению
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Создание выходного изображения
    smoothed_image = np.zeros_like(image)

    # Свертка изображения с фильтром для каждого канала
    for channel in range(channels):
        for i in range(height):
            for j in range(width):
                # Вычисление медианы в окне фильтра
                smoothed_image[i, j, channel] = np.median(padded_image[i:i + n, j:j + n, channel])

    return smoothed_image


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
        self.smoothing.clicked.connect(self.set_changes)
        self.gaussian.clicked.connect(self.set_changes)
        self.sigma_filter.clicked.connect(self.set_changes)
        self.diff.clicked.connect(self.set_changes)

    def load_image(self):
        filename = QFileDialog.getOpenFileName(self, "Загрузка изображения", "", "Image (*.png *.tiff *.bmp)")
        if filename[0] == "":
            QMessageBox.about(self, "Ошибка", "Файл не выбран")
            return
        filepath = filename[0]
        self.img = Image.open(filepath)
        # Преобразование изображения в массив numpy
        img_array = np.flipud(np.rot90(np.array(self.img)))
        self.img = img_array.astype(np.int32)
        self.img_original = copy.deepcopy(self.img)
        self.img_height = img_array.shape[1]
        self.img_width = img_array.shape[0]
        self.set_image()
        print(self.img.dtype)

    def save_image(self):
        if self.img is None:
            QMessageBox.about(self, "Ошибка", "Нечего сохранять")
            return
        filename = QFileDialog.getSaveFileName(self, "Open Image", "hue", "Image Files (*.png *.tiff *.bmp)")
        if filename[0] == "":
            QMessageBox.about(self, "Ошибка", "Путь сохранения не выбран")
            return
        self.image_view.getImageItem().save(filename[0])

    def set_changes(self):
        self.img = copy.deepcopy(self.img_original)
        if self.smoothing.isChecked():
            if self.nxn.currentText() == "3x3":
                n = 3
            else:
                n = 5
            if self.filter.currentText() == "Медианный фильтр":
                self.img = median_filter(self.img, n)
            else:
                self.img = mean_filter(self.img, n)
        if self.gaussian.isChecked():
            self.img = gaussian_filter(self.img, self.sigma.value())
        if self.sigma_filter.isChecked():
            self.img = sigma_filter(self.img, self.sigma_2.value())
        if self.diff.isChecked():
            self.img[:, :, :3] = np.abs(self.img - self.img_original)[:, :, :3]
        self.set_image()

    def set_image(self):
        self.image_view.clear()
        self.image_view.setImage(self.img)


if __name__ == "__main__":
    application = QtWidgets.QApplication(argv)
    program = Redactor()
    program.show()
    exit(application.exec_())
