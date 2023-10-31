from PyQt6 import QtCore, QtGui, QtWidgets
from datetime import date
import cv2
import numpy as np
import sys

import emotiondisplay

todayDate = date.today()
todayDateString = todayDate.strftime("%b-%d-%Y")

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(800, 600)
        self.pushButton = QtWidgets.QPushButton(parent=Form)
        self.pushButton.setGeometry(QtCore.QRect(320, 500, 200, 64))
        self.pushButton.setObjectName("pushButton")
        self.textBrowser_2 = QtWidgets.QTextBrowser(parent=Form)
        self.textBrowser_2.setGeometry(QtCore.QRect(80, 20, 641, 81))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textEdit = QtWidgets.QTextEdit(parent=Form)
        self.textEdit.setGeometry(QtCore.QRect(460, 200, 321, 281))
        self.textEdit.setProperty("userInput", "")
        self.textEdit.setObjectName("textEdit")
        self.textBrowser = QtWidgets.QTextBrowser(parent=Form)
        self.textBrowser.setGeometry(QtCore.QRect(460, 120, 321, 61))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setText(todayDateString)
        self.textBrowser.setStyleSheet("font-size: 24pt;")
        self.graphicsView = QtWidgets.QGraphicsView(parent=Form)
        self.graphicsView.setGeometry(QtCore.QRect(20, 120, 400, 361))
        self.graphicsView.setObjectName("graphicsView")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.pushButton.clicked.connect(self.capture_screen)
        self.update_graphics_view()

        self.timer = QtCore.QTimer(Form)
        self.timer.timeout.connect(self.update_graphics_view)
        self.timer.start(1000)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Emotion Detector Application"))
        self.pushButton.setText(_translate("Form", "Snapshot"))
        self.textBrowser_2.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                      "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                                      "p, li { white-space: pre-wrap; }\n"
                                                      "hr { height: 1px; border-width: 0; }\n"
                                                      "li.unchecked::marker { content: \"\\2610\"; }\n"
                                                      "li.checked::marker { content: \"\\2612\"; }\n"
                                                      "</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:26pt; font-weight:400; font-style:normal;\">\n"
                                                      "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:32pt;\">Emotion Detector Application</span></p></body></html>"))
        self.textEdit.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                 "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                                 "p, li { white-space: pre-wrap; }\n"
                                                 "hr { height: 1px; border-width: 0; }\n"
                                                 "li.unchecked::marker { content: \"\\2610\"; }\n"
                                                 "li.checked::marker { content: \"\\2612\"; }\n"
                                                 "</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:26pt; font-weight:400; font-style:normal;\">\n"
                                                 "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p></body></html>"))
        self.textEdit.setPlaceholderText("Type here...")

    def capture_screen(self):
        filename = f"screenshot_{todayDate}.png"
        screenshot = QtWidgets.QApplication.primaryScreen().grabWindow(Form.winId())
        screenshot.save(filename, "PNG")
        print("Screenshot saved")

    def update_graphics_view(self):
        result_image = emotiondisplay.get_emotion_detection_result()

        if result_image is not None:
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            height, width, channel = result_image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(result_image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)

            pixmap = QtGui.QPixmap.fromImage(q_image)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene = QtWidgets.QGraphicsScene()
            scene.addItem(item)
            self.graphicsView.setScene(scene)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec())
