import tkinter as tk
import pygubu  # pip install pygubu, designer installed with pip install pygubu-designer
import cv2

from Application.CameraType import CameraType
from Application.ImageType import ImageType
from PIL import Image
from PIL import ImageTk
from datetime import datetime
import os
import sys
from CNN.ExpressionRecognition import ExpressionRecognition


def fillCombobox(combobox, path):
    """
    TODO
    :param combobox:
    :param path:
    :return:
    """
    for file in os.listdir(path):
        if file not in combobox['values']:
            combobox['values'] = (*combobox['values'], file)


class GUI:

    def __init__(self):
        self.imageType = ImageType()
        self.cameraType = CameraType()  # start webcam rendering
        self.expressionrec = ExpressionRecognition()
        # Create pygubu builder
        self.builder = builder = pygubu.Builder()

        # Load .ui File
        builder.add_from_file('./UserInterface/facialdetectionGUI.ui')

        # set the mainwindow to the toplevel object
        self.mainwindow = builder.get_object('toplevel')
        self.mainwindow.protocol("WM_DELETE_WINDOW", self.closeWindow)
        # get imgLabel for displaying loaded images
        self.imgLabel = builder.get_object('imgLabel')
        # latest image form imagemode
        self.imageInImageMode = None

        self.cnnBar = builder.get_object('comboboxCNN')
        self.haarBar = builder.get_object('comboboxHaar')
        # Fill CNN Model Combobox
        fillCombobox(self.cnnBar, "./Resources/Models/")
        # Fill Haarcascade Combobox
        fillCombobox(self.haarBar, "./Resources/Haarcascade/")
        #Set Default Image for Imagemode
        self.setImage("./Resources/Default/default_image.png")

        ## Camera Mode Selection ##
        self.boxCameraSelection = builder.get_object('comboboxCameraSelect')
        self.boxCameraSelection.current(0)


        # Connect Callback Functions - ButtonClicks etc
        builder.connect_callbacks(self)

    def closeWindow(self):
        """
        TODO
        :return:
        """
        #self.mainwindow.quit()
        self.mainwindow.destroy()
        sys.exit("Shut down system")

    def on_load_cnn_model_button_click(self):
        """
        TODO
        :return:
        """
        cnnModel = self.cnnBar.get()
        if cnnModel != "":
            self.expressionrec.updateModel(cnnModel)
            print("SUCCESS: Loaded a new CNN Model and adjusted parameters")
        else:
            print("ERROR: No File selected when loading new Haarcascade")

    def on_load_haar_model_button_click(self):
        """
        TODO
        :return:
        """
        classifier = self.haarBar.get()
        if classifier != "":
            self.imageType.updateCascadeClass(classifier)
            print("SUCCESS: Loaded a new Haarcascade for Facedetection")
        else:
            print("ERROR: No File selected when loading new Haarcascade")

    def on_start_camera_button_click(self):
        cameratype = self.boxCameraSelection.get()
        if cameratype != "":
            print("SUCCESS: Start Camera")
            self.cameraType.runCamera(cameratype)
        else:
            print("ERROR: Camera already running or no camera selected")


    def on_save_image_from_ImageMode_button_click(self):
        """
        This Method gets called, when the Save Button in the Image Panel is clicked.
        The latest FER picture will be saved in the ./Screenshot/LatestFace directory as .jpg file using
        a date- and time-stamp.
        :return: Nothing
        """
        if self.imageInImageMode is not None:
            cv2.imwrite('./Screenshot/LatestFace' + datetime.now().strftime("%m%d%Y,%H%M%S") + '.jpg',
                        self.imageInImageMode)
            print("SUCCESS: Saved image from ImageMode in the directory ./Screenshot/")

    def on_load_image_button_click(self):
        """
        When loading a picture, a filedialog will be openend, where you can select a proper image
        :return: Nothing
        """

        path = tk.filedialog.askopenfilename()

        if not (path.endswith('.jpeg') or path.endswith('.jpg') or path.endswith('.JPG') or path.endswith(
                '.png') or path.endswith('.PNG')):
            print("You did not select a Image with correct Format, only use jpeg,jpg or png!")
            return
        # Selected a correct format
        self.setImage(path)

    def setImage(self, path):
        """
        TODO
        :param path:
        :return:
        """
        image = self.imageType.runImage(path, self.expressionrec)
        if image is not None:
            self.imageInImageMode = image
            # for setting image on label, we have to convert into a fitting format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # from bgr to rgb
            image = Image.fromarray(image)  # creates image from array
            image = ImageTk.PhotoImage(image)  # convert for tkinter
            # update the panel
            self.imgLabel.configure(image=image)
            self.imgLabel.image = image

    def run(self):
        """

        :return:
        """
        self.mainwindow.mainloop()
