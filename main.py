import copy
import itertools
import random
import warnings

from matplotlib import pyplot as plt
from numba import njit
from pyqtgraph import SignalProxy

warnings.filterwarnings('ignore')
from sys import argv
import numpy as np
import cv2 as cv
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5 import uic
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
# from numba import njit
Ui_MainWindow, _ = uic.loadUiType("interface_lab_2.ui")




class Redactor(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Redactor, self).__init__()
        self.setupUi(self)
        self.img = None
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        sel
        self.load_image_action.triggered.connect(self.load_image)
        self.save_image_action.triggered.connect(self.save_image)

    # Функция для обработки видео с использованием алгоритма Хорна-Шанка
    def horn_schunck(self, video_path):
        cap = cv.VideoCapture(video_path)
        # Инициализация параметров алгоритма Хорна-Шанка
        params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # Инициализация предыдущего кадра
        ret, prev_frame = cap.read()
        prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        # Чтение видео и применение алгоритма Хорна-Шанка к каждому кадру
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Применение алгоритма Хорна-Шанка к двум последовательным кадрам
            flow = cv.calcOpticalFlowFarneback(prev_frame, frame_gray, None, **params)
            # Визуализация потока
            # здесь можно добавить код для визуализации потока на видео
            # например, отрисовать векторы потока на кадре
            # или использовать стрелки для указания направления движения объектов
            # обновление предыдущего кадра
            prev_frame = frame_gray
            # показать текущий кадр с обнаруженным оптическим потоком
            cv.imshow('Optical Flow - Horn-Schunck', frame)
            if cv.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    def lucas_kanade(self,video_path):
        cap = cv.VideoCapture(video_path)
        # Инициализация параметров алгоритма Лукаса-Канаде
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # Инициализация цвета для отрисовки треков объектов
        color = (0, 255, 0)
        # Чтение первого кадра
        ret, old_frame = cap.read()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        # Определение начальных точек для отслеживания (можно изменить на ваши нужды)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7,
                                                                blockSize=7))
        # Создание пустого массива для хранения результатов
        mask = np.zeros_like(old_frame)
        # Чтение видео и применение алгоритма Лукаса-Канаде к каждому кадру
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Применение алгоритма Лукаса-Канаде
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Выбор хороших точек
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # Отрисовка треков на кадре
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (a, b), (c, d), color, 2)
                frame = cv.circle(frame, (a, b), 5, color, -1)
            # Отображение результатов
            img = cv.add(frame, mask)
            cv.imshow('Optical Flow - Lucas-Kanade', img)
            # Обновление кадра и точек для следующей итерации
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            # Выход из цикла по нажатию клавиши 'q'
            if cv.waitKey(30) & 0xFF == ord('q'):
                break
        # Освобождение ресурсов и закрытие окон
        cap.release()
        cv.destroyAllWindows()

    def load_image(self):
        filename = QFileDialog.getOpenFileName(self, "Загрузка изображения", "", "Image (*.png *.tiff *.bmp)")
        if filename[0] == "":
            QMessageBox.about(self, "Ошибка", "Файл не выбран")
            return
        filepath = filename[0]
        self.img = Image.open(filepath)
        if self.img.mode == 'RGBA':
            self.img = self.img.convert('RGB')
        img_array = np.flipud(np.rot90(np.array(self.img)))
        self.img = img_array.astype(np.int32)
        self.img_original = copy.deepcopy(self.img)
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


    def set_image(self):
        self.image_view.clear()
        self.image_view.setImage(self.img)


if __name__ == "__main__":
    application = QtWidgets.QApplication(argv)
    program = Redactor()
    program.show()
    exit(application.exec_())
