import copy
import warnings

warnings.filterwarnings('ignore')
from sys import argv
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5 import uic
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

Ui_MainWindow, _ = uic.loadUiType("interface_lab_2.ui")


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
        self.segment_rgb_action.clicked.connect(self.segment_rgb)
        self.segment_lab_action.clicked.connect(self.segment_lab)
        self.dbrgb.clicked.connect(self.db_rgb)
        self.dblab.clicked.connect(self.db_lab)

    def segment_rgb(self):
        self.img = self.img_original.copy()
        self.img = self.apply_clustering_rgb()
        self.set_image()

    def segment_lab(self):
        self.img = self.img_original.copy()
        self.img = self.apply_clustering_lab()
        self.set_image()

    def db_lab(self):
        self.img = self.img_original.copy()
        self.img = self.dbscan(True)
        self.set_image()

    def db_rgb(self):
        self.img = self.img_original.copy()
        self.img = self.dbscan()
        self.set_image()

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

    def apply_clustering_lab(self):

        lab_img = self.rgb_to_lab(self.img)
        reshaped_lab_img = lab_img.reshape(-1, 3)

        # Apply k-means clustering in Lab space
        kmeans_lab = KMeans(n_clusters=self.kVal.value(), random_state=42)
        lab_labels = kmeans_lab.fit_predict(reshaped_lab_img)
        segmented_img_lab = kmeans_lab.cluster_centers_[lab_labels].reshape(self.img.shape)

        return segmented_img_lab.astype(np.uint8)

    def apply_clustering_rgb(self):
        # Reshape image array for clustering
        reshaped_img = self.img.reshape(-1, 3)

        # Standardize features
        scaler = StandardScaler()
        standardized_img = scaler.fit_transform(reshaped_img)

        # Apply k-means clustering in RGB space
        kmeans_rgb = KMeans(n_clusters=self.kVal.value(), random_state=42)
        rgb_labels = kmeans_rgb.fit_predict(standardized_img)

        # Generate random colors for each cluster
        cluster_colors = [tuple(np.random.randint(0, 256, size=3)) for _ in range(self.kVal.value())]

        # Assign random colors to each pixel based on cluster labels
        segmented_img_rgb = np.array([cluster_colors[label] for label in rgb_labels])
        segmented_img_rgb = segmented_img_rgb.reshape(self.img.shape)

        return segmented_img_rgb.astype(np.uint8)

    def dbscan(self, lab=False):
        if lab:
            self.img = self.rgb_to_lab(self.img)
        # Reshape image array for clustering
        reshaped_img = self.img.reshape(-1, 3)

        # Standardize features
        scaler = StandardScaler()
        standardized_img = scaler.fit_transform(reshaped_img)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps.value(), min_samples=self.samples.value())
        dbscan_labels = dbscan.fit_predict(standardized_img)

        # Generate random colors for each cluster
        unique_labels = np.unique(dbscan_labels)
        cluster_colors = [tuple(np.random.randint(0, 256, size=3)) for _ in range(len(unique_labels))]
        # Assign random colors to each pixel based on cluster labels
        segmented_img_dbscan = np.array([cluster_colors[label] for label in dbscan_labels])
        segmented_img_dbscan = segmented_img_dbscan.reshape(self.img.shape)

        return segmented_img_dbscan.astype(np.uint8)

    def rgb_to_lab(self, rgb_img):
        rgb_img = rgb_img.astype(np.uint8)
        pil_img = Image.fromarray(rgb_img)
        lab_img = pil_img.convert('LAB')
        lab_img = np.array(lab_img)
        return lab_img

    def set_image(self):
        self.image_view.clear()
        self.image_view.setImage(self.img)


if __name__ == "__main__":
    application = QtWidgets.QApplication(argv)
    program = Redactor()
    program.show()
    exit(application.exec_())
