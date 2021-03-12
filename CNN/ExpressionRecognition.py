"""ExpressionRecognition.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Description**: Bachelorthesis - Prototype for FER

**Institution**: Johannes Kepler University Linz - Institute of Computational Perception

This class uses a trained model (.h5 file) for FER. Given a current face image, detected by a haarcascade, The image
will be first resized for fitting into the selected model and then the output prediction gets maximized and the emotion
will be returned as a string. All models loaded should diver 7 main emotions:
'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise' - in that indexing order.

**Required installations for running this class**:
OpenCV: pip install opencv-python       https://pypi.org/project/opencv-python/
Tensorflow: pip install tensorflow      https://www.tensorflow.org/
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
        """
        This method gets called, when a new model should be used for emotion detection.
        Depending on the naming of the file, we can select the proper amount of input channels and pixelsizes.
        This method is only dealing with p100 or p48 and dim1 or dim3.
        :param modelpath: the name of the .h5 file to be selected.
        :return: Nothing
        """
        self.modelkeras = "./Resources/Models/" + modelpath
        self.loaded_model = tf.keras.models.load_model(self.modelkeras)
        # Now imageprocessor parameters have to be changed apparently
        self.inputdim = self.loaded_model.input_shape[3]
        self.imgsize = self.loaded_model.input_shape[2]

    def imagepreprocessor(self, img):
        """
        The Imagepreprocessor is used for reshaping the input image and doing histogramm equal. if model dim. is of
        channelsize 1.
        :param img: original image to be predicted
        :return: normalized picture, fitting for the chosen model
        """
        if self.inputdim == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)

        img = cv2.resize(img, (self.imgsize, self.imgsize))

        if self.inputdim == 1:
            img = np.array(img).reshape(self.imgsize, self.imgsize, self.inputdim)
        try: # normaly you should not use try for control flow.
            self.loaded_model.get_layer("tf.math.truediv")  # if model already implies division, dont do it manually
        except:
            img = img / 255.0
        return img

    def getEmotion(self, faceImage):
        """
        This method will return a emotion as a string for a given faceImage.
        :param faceImage: image to be predicted
        :return: emotion string with maximum confidence.
        """
        temp = []  # reshape testdata, becuase predict function needs 4 dimensions
        img = self.imagepreprocessor(faceImage)
        temp.append(img)
        temp = np.array(temp)
        predictions = self.loaded_model.predict(temp)
        return self.emotions[np.argmax(predictions)]
