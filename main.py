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
        self.penis_button.clicked.connect(lambda: self.farenbeck(r"C:\Users\user\Downloads\videoplayback.mp4"))
        self.vagina_button.clicked.connect(lambda: self.horn_schunck(r"C:\Users\user\Downloads\videoplayback.mp4"))
        self.zhopa_buton.clicked.connect(lambda: self.lucas_kanade(r"C:\Users\user\Downloads\videoplayback.mp4"))

        self.load_image_action.triggered.connect(self.load_image)
        self.save_image_action.triggered.connect(self.save_image)

    # Функция для обработки видео с использованием алгоритма Хорна-Шанка
    def farenbeck(self, video_path):
        cap = cv.VideoCapture(video_path)
        # Инициализация параметров алгоритма Фарнебэка
        params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # Инициализация предыдущего кадра
        ret, prev_frame = cap.read()
        if not ret:
            print("Ошибка чтения видео.")
            return
        prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        # Чтение видео и применение алгоритма Фарнебэка к каждому кадру
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Применение алгоритма Фарнебэка к двум последовательным кадрам
            flow = cv.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, **params)
            # Визуализация оптического потока
            hsv = np.zeros_like(prev_frame)
            hsv[..., 1] = 255

            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            # Показать текущий кадр с обнаруженным оптическим потоком
            cv.imshow('Optical Flow - Farneback', bgr)

            prev_frame_gray = frame_gray

            if cv.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def horn_schunck(self, video_path):
        def compute_horn_schunck(I1, I2, alpha=0.001, num_iterations=100):
            I1 = I1.astype(np.float32) / 255.0
            I2 = I2.astype(np.float32) / 255.0

            Ix = cv.Sobel(I1, cv.CV_64F, 1, 0, ksize=5)
            Iy = cv.Sobel(I1, cv.CV_64F, 0, 1, ksize=5)
            It = I2 - I1

            u = np.zeros_like(I1)
            v = np.zeros_like(I1)

            kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                               [1 / 6, 0, 1 / 6],
                               [1 / 12, 1 / 6, 1 / 12]])

            for _ in range(num_iterations):
                u_avg = cv.filter2D(u, -1, kernel)
                v_avg = cv.filter2D(v, -1, kernel)

                P = Ix * u_avg + Iy * v_avg + It
                D = alpha ** 2 + Ix ** 2 + Iy ** 2

                u = u_avg - (Ix * P) / D
                v = v_avg - (Iy * P) / D

            return u, v

        cap = cv.VideoCapture(video_path)

        ret, prev_frame = cap.read()
        if not ret:
            print("Ошибка чтения видео.")
            return

        prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            u, v = compute_horn_schunck(prev_frame_gray, frame_gray)

            hsv = np.zeros_like(prev_frame)
            hsv[..., 1] = 255

            mag, ang = cv.cartToPolar(u, v)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            cv.imshow('Optical Flow - Horn-Schunck', bgr)

            prev_frame_gray = frame_gray

            if cv.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def lucas_kanade(self,video_path):
        cap = cv.VideoCapture(video_path)

        # Параметры для функции оптического потока Лукаса-Канаде
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Считывание первого кадра и его преобразование в градации серого
        ret, old_frame = cap.read()
        if not ret:
            print("Ошибка чтения видео.")
            return
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        # Инициализация точек для отслеживания (например, углы по методу Ши-Томаси)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None,
                                    **dict(maxCorners=200, qualityLevel=0.3, minDistance=5, blockSize=15))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Вычисление оптического потока по методу Лукаса-Канаде
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Выбор хороших точек
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Создание новой маски для каждого кадра, чтобы отображать линии только для двух ближайших кадров
                mask = np.zeros_like(frame)

                # Рисование треков
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                img = cv.add(frame, mask)

                cv.imshow('Optical Flow - Lucas-Kanade', img)

                # Обновление предыдущего кадра и точек
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

            if cv.waitKey(30) & 0xFF == ord('q'):
                break

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
