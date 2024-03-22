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


def sigma_filter(image, sigma):
    """
    Applies Gaussian smoothing to an image.

    Args:
        image: np.array representing the image.
        sigma: The sigma value for the Gaussian kernel.

    Returns:
        np.array, the smoothed image.
    """

    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)  # Ensure odd size
    x, y = np.mgrid[-kernel_size // 2 + 1: kernel_size // 2 + 1, -kernel_size // 2 + 1: kernel_size // 2 + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # Normalize

    # Convolve image with kernel
    smoothed_image = convolve2d(image, kernel, mode='same')

    # Optional normalization (adjust as needed)
    smoothed_image = np.clip(smoothed_image, 0, 255).astype(np.uint8)

    return smoothed_image

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

    def log_transform(self):
        c_r = 255 / np.log(1 + np.max(self.img[:, :, 0]))
        c_g = 255 / np.log(1 + np.max(self.img[:, :, 1]))
        c_b = 255 / np.log(1 + np.max(self.img[:, :, 2]))
        transformed_r = c_r * np.log(1 + np.where(self.img[:, :, 0] == 255, 254, self.img[:, :, 0]))
        transformed_g = c_g * np.log(1 + np.where(self.img[:, :, 1] == 255, 254, self.img[:, :, 1]))
        transformed_b = c_b * np.log(1 + np.where(self.img[:, :, 2] == 255, 254, self.img[:, :, 2]))
        transformed_img = np.stack((transformed_r, transformed_g, transformed_b), axis=-1)
        transformed_img = np.clip(transformed_img, 0, 255).astype(np.uint8)
        alpha = self.img[:, :, 3]
        transformed_image_rgba = np.dstack((transformed_img, alpha))
        self.img = transformed_image_rgba
        self.set_image()

    def power_transform(self):
        gamma = 0.5
        r_max = np.max(self.img[:, :, 0])
        g_max = np.max(self.img[:, :, 1])
        b_max = np.max(self.img[:, :, 2])
        desired_max = 100
        c_r = desired_max / (r_max ** gamma)
        c_g = desired_max / (g_max ** gamma)
        c_b = desired_max / (b_max ** gamma)
        transformed_r = c_r * (self.img[:, :, 0] ** gamma)
        transformed_g = c_g * (self.img[:, :, 1] ** gamma)
        transformed_b = c_b * (self.img[:, :, 2] ** gamma)
        transformed_r = np.clip(transformed_r, 0, 255).astype(np.uint8)
        transformed_g = np.clip(transformed_g, 0, 255).astype(np.uint8)
        transformed_b = np.clip(transformed_b, 0, 255).astype(np.uint8)
        transformed_img = np.stack((transformed_r, transformed_g, transformed_b, self.img[:, :, 3]), axis=-1)
        self.img = transformed_img
        self.set_image()

    def binarization(self):
        threshold = 65
        brightness = (self.img[:, :, 0] + self.img[:, :, 1] + self.img[:, :, 2]) / 3
        binary_img = np.where(brightness > threshold, 255, 0)
        binary_img_rgba = np.stack((binary_img, binary_img, binary_img, self.img[:, :, 3]), axis=-1)
        self.img = binary_img_rgba
        self.set_image()

    def goofy_ahh_pixel_cutting(self):
        img_array = self.img
        # Разделяем RGB и альфа-каналы
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]
        min_brightness = 10
        max_brightness = 100
        constant_value = 0
        mask = (rgb >= min_brightness) & (rgb <= max_brightness)

        if constant_value is not None:
            rgb[~mask] = constant_value
            processed_image_array = np.concatenate((rgb, alpha[:, :, np.newaxis]), axis=2)
        else:
            processed_image_array = img_array
        self.img = processed_image_array
        self.set_image()

    def sobel_operator(self):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        gray = np.dot(self.img[:, :, :3], [0.299, 0.587, 0.114])

        grad_x = convolve2d(gray, sobel_x, mode="same")
        grad_y = convolve2d(gray, sobel_y, mode="same")
        abs_grad_sum = np.abs(grad_x) + np.abs(grad_y)
        sharpness_score = np.mean(abs_grad_sum)
        print(sharpness_score)

    def sharpness_score(self):
        differences = []
        for r in range(1, self.img_width - 1):
            for c in range(1, self.img_height - 1):
                pixel_brightness = self.img[r][c][0:3].mean()
                neighbour1 = self.img[r - 1][c][0:3].mean()
                neighbour2 = self.img[r][c + 1][0:3].mean()
                neighbour3 = self.img[r][c - 1][0:3].mean()
                neighbour4 = self.img[r + 1][c][0:3].mean()
                differences.append(abs(neighbour1 - pixel_brightness))
                differences.append(abs(neighbour2 - pixel_brightness))
                differences.append(abs(neighbour3 - pixel_brightness))
                differences.append(abs(neighbour4 - pixel_brightness))
        print(sum(differences) / len(differences))


if __name__ == "__main__":
    application = QtWidgets.QApplication(argv)
    program = Redactor()
    program.show()
    exit(application.exec_())
