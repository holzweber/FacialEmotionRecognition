"""UserInterface.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Description**: Bachelorthesis - Prototype for FER

**Institution**: Johannes Kepler University Linz - Institute of Computational Perception

This file handles the general userinterface creation and coordination of the userinput.
Since the userinterface is rather small, most of the features will be self explaining.
The user is able to experiment with own trained models, both for FER and facedetection.

**Required installations for running this class**:
Pygubu: pip install pygubu              https://pypi.org/project/pygubu/
        pip install pygubu-designer
OpenCV: pip install opencv-python       https://pypi.org/project/opencv-python/
"""
# User Interface Imports
import tkinter.filedialog
import tkinter.messagebox
import tkinter as tk  # comes with python installer python.org
import pygubu  # pip install pygubu, designer installed with pip install pygubu-designer
# File and Image - Handling Imports
import cv2  # OpenCV for part of the Image Handling
from PIL import Image, ImageTk  # pip install Pillow
from datetime import datetime
import os
# FER Imports
from Application.CameraMode import CameraMode
from Application.ImageMode import ImageMode
from CNN.ExpressionRecognition import ExpressionRecognition
import matplotlib.pyplot as plt


def fillCombobox(combobox, path):
    """
    The fill Combox method is used for the CNN Model and Haarcascade Combobox.
    When putting new files int the ./Resources/Models or ./Resources/Haarcascade directorys,
    also this files will be made available for FER.
    :param combobox: The combox which will be filled with files
    :param path: One of the two directorys in ./Resources/*
    :return: Nothing
    """
    for file in os.listdir(path):
        if file not in combobox['values']:
            combobox['values'] = (*combobox['values'], file)


def showImageInLabel(imgdir, label):
    """
    This method will show a given image from 'dir' in the given label 'label
    :param imgdir: path to image
    :param label: label to be used
    :return:
    """
    image = cv2.imread(imgdir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # from bgr to rgb
    image = Image.fromarray(image)  # creates image from array
    image = ImageTk.PhotoImage(image)  # convert for tkinter
    # Update the Imagepanel
    label.configure(image=image)
    label.image = image


class GUI:

    def __init__(self):
        """
        Set default values of the project and create instances of the ImageMode, CameraMode and ExpressionRecognition
        classes. Also the Userinterface will be loaded and default values, callbacks will be set.
        """
        self.imageMode = ImageMode()
        self.cameraMode = CameraMode()  # start webcam rendering
        self.expressionrec = ExpressionRecognition()

        # Create pygubu builder

        self.builder = pygubu.Builder()

        # Load .ui File
        self.builder.add_from_file('./UserInterface/fer_userinterface.ui')

        # set the mainwindow to the toplevel object
        self.mainwindow = self.builder.get_object('toplevel')
        self.mainwindow.protocol("WM_DELETE_WINDOW", self.closeWindow)

        # get imgLabel for displaying loaded images
        self.imgLabel = self.builder.get_object('imgLabel')
        # latest image form imagemode
        self.imageInImageMode = None
        self.statimg = None  # holds image with accuracy for advanced image mode

        self.cnnBar = self.builder.get_object('comboboxCNN')
        self.haarBar = self.builder.get_object('comboboxHaar')

        # Fill CNN Model Combobox
        fillCombobox(self.cnnBar, "./Resources/Models/")
        # Fill Haarcascade Combobox
        fillCombobox(self.haarBar, "./Resources/Haarcascade/")
        # Set Default Image for Imagemode

        # Camera Mode Selection
        self.boxCameraSelection = self.builder.get_object('comboboxCameraSelect')
        self.boxCameraSelection.current(0)  # set Default Camera to 'Internal'

        # Camera Statistik Vector
        self.latestCamStat = None
        self.labelCamStatistik = self.builder.get_object('labelCamStatistik')
        self.labelCamStatistik.bind('<Double-Button-1>', self.on_click_close_stat)
        # Connect Callback Functions - ButtonClicks etc
        self.builder.connect_callbacks(self)

    def closeWindow(self):
        """
        This method gets called, when the user closed the Userinterface.
        Here the mainwindow will be destroyed.
        :return: Nothing
        """
        self.mainwindow.destroy()
        self.mainwindow.quit()

    def on_press_image_science_mode(self):
        """
        This method will activate a prediction describtion for each encountered face in the loaded image
        :return:
        """
        if self.statimg is not None:
            self.imageInImageMode = self.statimg
            # for setting image on label, we have to convert into a fitting format
            self.imageInImageMode = cv2.cvtColor(self.imageInImageMode, cv2.COLOR_BGR2RGB)  # from bgr to rgb
            self.imageInImageMode = Image.fromarray(self.imageInImageMode)  # creates image from array
            self.imageInImageMode = ImageTk.PhotoImage(self.imageInImageMode)  # convert for tkinter
            # update the panel
            self.imgLabel.configure(image=self.imageInImageMode)
            self.imgLabel.image = self.imageInImageMode
        else:
            tk.messagebox.showerror(title="Show Accuracy Error",
                                    message="No image loaded!")

    def on_click_close_stat(self, event):
        """
        This method gets called, when double clicking on the displayed statistic, in order to close it again
        :param event: Not used, but needed as parameter
        :return: Nothing
        """
        self.labelCamStatistik.configure(image=None)
        self.labelCamStatistik.image = None

    def on_click_show_camera_statistics(self):
        """
        This method show the latest statistics of the camera mode and also save the diagram in the
        Statistics Folder
        :return: Nothing
        """
        if self.latestCamStat is not None:
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

            plt.bar(emotions,
                    self.latestCamStat,
                    align='center',
                    alpha=0.5,
                    color=['red', 'green', 'cyan', 'magenta', 'blue', 'yellow', 'black'])
            plt.xlabel("Emotion")
            plt.ylabel("Rel. Freq [%]")
            plt.title("Latest Detected Emotions")
            storedir = "./Statistics/LatestFace" + datetime.now().strftime("%m%d%Y-%H%M%S") + ".jpg"
            plt.savefig(storedir, dpi=70)
            plt.close() # close plt to avoid overlayers
            # Show image
            showImageInLabel(storedir, self.labelCamStatistik)
        else:
            tk.messagebox.showerror(title="Camera Statistic Error",
                                    message="You did not run the camera!")

    def on_load_cnn_model_button_click(self):
        """
        This method will get called, when the user has chosen a new CNN Model for FER and then pressed
        the "Load" button. After that, the updateModel(model) method of the ExpressionRecognition object
        will get called, where the CNN Model will be updated, for the next FER
        :return: Nothing
        """
        cnnModel = self.cnnBar.get()
        if cnnModel != "" and cnnModel.endswith('.h5'):
            self.expressionrec.updateModel(cnnModel)
            print("SUCCESS: Loaded a new CNN Model and adjusted parameters")
            tk.messagebox.showinfo(title="Update CNN Model", message="Loaded new CNN Model and adjusted parameters")
        else:
            tk.messagebox.showerror(title="CNN Model Error", message="No CNN file selected, or wrong fileformat!")
            print("ERROR: No file selected when loading new haarcascade or wrong fileformat (has to be .h5)! ")

    def on_load_haar_model_button_click(self):
        """
        This method will get called, when the user has chosen a new Haarcascade for Face Detection and then pressed
        the "Load" button. After that, the updateCascadeClass(classifier) method of both, the ImageMode and the
        CameraMode object will get called, where the Face Detector will be updated, for the next FER.
        :return: Nothing
        """
        classifier = self.haarBar.get()
        if classifier != "" and classifier.endswith('.xml'):
            self.imageMode.updateCascadeClass(classifier)
            self.cameraMode.updateCascadeClass(classifier)
            tk.messagebox.showinfo(title="Update Haarcascade",
                                   message="Loaded new Haarcascade and adjusted parameters")
            print("SUCCESS: Loaded a new Haarcascade for facedetection")
        else:
            tk.messagebox.showerror(title="Haarcascade Error",
                                    message="No Haarcascade file selected, or wrong fileformat!")
            print("ERROR: No file selected when loading new haarcascade or wrong fileformat (has to be .xml)!")

    def on_start_camera_button_click(self):
        """
        This method starts the selected camera in a new Window. Until closing the cameraWindow, the Userinterface
        will be blocked.
        :return: Nothing
        """
        cameratype = self.boxCameraSelection.get()
        if cameratype != "":
            tk.messagebox.showinfo(title="Start Camera", message="Press 'f' to store face.\n"
                                                                 "Press 'c' to take screenshot.\n"
                                                                 "Press 'q' to turn off camera.")

            print("SUCCESS: Start Camera - userinterface blocked until closing Camera Window")
            self.latestCamStat = None
            self.latestCamStat = self.cameraMode.runCamera(cameratype, self.expressionrec)
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
            cv2.imwrite('./Screenshot/LatestFace' + datetime.now().strftime("%m%d%Y,%H%M") + '.jpg',
                        self.imageInImageMode)
            tk.messagebox.showinfo(title="Save Image", message="Stored Image in ./LatestFace directory")
            print("SUCCESS: Saved image from ImageMode in the directory ./Screenshot/")

    def on_load_image_button_click(self):
        """
        When loading a picture, a filedialog will be pop up for selecting a picture from the local directory.
        Allowed formats are: .jpeg,.JPG,.jpg,.png,.PNG
        Facial Emotions on the pictures will be automatically recognized and then loaded in the userinterface
        :return: Nothing
        """

        path = tk.filedialog.askopenfilename()  # select image to be predicted

        if not (path.endswith('.jpeg') or path.endswith('.jpg') or path.endswith('.JPG') or path.endswith(
                '.png') or path.endswith('.PNG')):
            tk.messagebox.showerror(title="Image Error",
                                    message="You did not select a Image with correct Format!")
            print("ERROR: You did not select a Image with correct Format, only use jpeg,jpg or png!")
            return
        # Selected a correct format

        self.setImage(path)

    def setImage(self, path):
        """
        The setImage method is used, to set the selected picture (and already FER) in the GUI.
        :param path: path of the choosen picture from a local directory.
        :return: Nothing
        """
        image, self.statimg = self.imageMode.runImage(path, self.expressionrec)

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
        This method starts the Userinterface, by opening the mainwindow.
        :return: Nothing
        """
        self.mainwindow.mainloop()
