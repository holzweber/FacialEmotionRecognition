"""ExpressionRecognition.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Matr.-Nr.**: k11803108

**Description**: Bachelorthesis - Prototype for FER

In this  File ...
TODO
"""


class ExpressionRecognition:

    def __init__(self):
        self.trainedModel = "./Resources/FER-p99.h5"
        self.Classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def getEmotion(self, faceImage):
        # return self.Classes[1]
        pass

