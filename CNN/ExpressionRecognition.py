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


def imgpreprocessor(img):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = cv2.equalizeHist(temp)
    temp = cv2.resize(temp, (48, 48))
    temp = np.array(temp).reshape(48,48,1)
    temp = temp / 255.0
    return temp


class ExpressionRecognition:

    def __init__(self):
        self.Classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.modelkeras = "model2020_fer2013_p48.h5"
        self.Classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def getEmotion(self, faceImage):
        temp = []  # reshape testdata, becuase predict function needs 4 dimensions
        img = imgpreprocessor(faceImage)
        temp.append(img)
        temp = np.array(temp)
        loaded_model = tf.keras.models.load_model("./Resources/" + self.modelkeras)
        predictions = loaded_model.predict(temp)
        return self.Classes[np.argmax(predictions)]
        pass
