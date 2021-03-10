"""ExpressionRecognition.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Matr.-Nr.**: k11803108

**Description**: Bachelorthesis - Prototype for FER

In this  File ...
TODO
"""
import tensorflow as tf  # import tensorflow module
import cv2  # library for image-handling
import numpy as np  # standard lib. for calculations


class ExpressionRecognition:

    def __init__(self):
        # possible emotion classes
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        # set default model
        self.modelkeras = "./Resources/Models/model2020_fer2013_p48_dim1.h5"
        # For ignoring tensorflow warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        # load default model
        self.loaded_model = tf.keras.models.load_model(self.modelkeras)
        # set default image sizes
        self.imgsize = 48
        self.inputdim = 1

    def updateModel(self, modelpath):
        self.modelkeras = "./Resources/Models/" + modelpath
        self.loaded_model = tf.keras.models.load_model(self.modelkeras)
        # Now imageprocessor parameters have to be changed apparently
        if "p48" in modelpath:
            self.imgsize = 48
        elif "p100" in modelpath:
            self.imgsize = 100
        if "dim1" in modelpath:
            self.inputdim = 1
        elif "dim3" in modelpath:
            self.inputdim = 3

    def imagepreprocessor(self, img):

        if self.inputdim == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
        img = cv2.resize(img, (self.imgsize, self.imgsize))
        if self.inputdim == 1:
            img = np.array(img).reshape(self.imgsize, self.imgsize, self.inputdim)
        img = img / 255.0
        return img

    def getEmotion(self, faceImage):
        temp = []  # reshape testdata, becuase predict function needs 4 dimensions
        img = self.imagepreprocessor(faceImage)
        temp.append(img)
        temp = np.array(temp)
        predictions = self.loaded_model.predict(temp)
        return self.emotions[np.argmax(predictions)]
