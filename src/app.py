
import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
class InpaintingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Object Removal'
        self.initUI()
        
       
        self.rcnn, self.transforms = self.load_rcnn_model()
        self.deepfill = self.load_deepfill_model()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('path/to/icon.png'))  

       
        app.setStyle('Fusion')

       
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

       
        self.btnBrowse = QPushButton('Browse Image', self)
        self.btnBrowse.setFont(QFont('Arial', 12))

        layout.addWidget(self.btnBrowse)


        self.label = QLabel(self)
        self.label.setStyleSheet("border: 1px solid black;")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        
        self.btnRun = QPushButton('Select Object', self)
        self.btnRun.setFont(QFont('Arial', 12))

        layout.addWidget(self.btnRun)

       
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    
        self.btnBrowse.clicked.connect(self.openFileNameDialog)
        self.btnRun.clicked.connect(self.run_inpainting)

        self.applyCustomStyle()

        self.show()

    def applyCustomStyle(self):
        
        button_style = """
        QPushButton {
            background-color: #89CFF0;
            border-style: outset;
            border-width: 2px;
            border-radius: 10px;
            border-color: beige;
            font: bold 14px;
            padding: 6px;
        }
        QPushButton:pressed {
            background-color: #5599FF;
            border-style: inset;
        }
        """
        self.btnBrowse.setStyleSheet(button_style)
        self.btnRun.setStyleSheet(button_style)

      
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        app.setPalette(palette)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if fileName:
            self.image_path = fileName
            pixmap = QPixmap(fileName)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def load_rcnn_model(self):
        print("Creating rcnn model")
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        transforms = weights.transforms()
        rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
        rcnn = rcnn.eval()
        return rcnn, transforms

    def load_deepfill_model(self):
        print('Creating deepfil model')
        for f in os.listdir('src/models'):
            if f.endswith('.pth'):
                deepfill_weights_path = os.path.join('src/models', f)
        deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)
        return deepfill

    def run_inpainting(self):
        if hasattr(self, 'image_path'):
            model = ObjectRemove(segmentModel=self.rcnn,
                                 rcnn_transforms=self.transforms,
                                 inpaintModel=self.deepfill,
                                 image_path=self.image_path)
            output = model.run()
            output_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imwrite('temp_output_image.png', output_image)
            pixmap = QPixmap('temp_output_image.png')
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            os.remove('temp_output_image.png')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = InpaintingApp()
    sys.exit(app.exec_())
