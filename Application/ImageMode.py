"""CameraMode.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Matr.-Nr.**: k11803108

**Description**: Bachelorthesis - Prototype for FER

In this File the Camera Prototype is running.
Taking a trained CNN for calssifing emotion of a choosen picture.
The picture has to be in the format of png,jpg or jpeg (watch out for ending of your file!)

Details on Usage:
TODO

Main Features of the Displaying and FaceTracking are done Using the OpenCV Library, which can be looked up
here: https://opencv.org/
"""
import tkinter
import tkinter.filedialog
import cv2
from CNN.ExpressionRecognition import ExpressionRecognition

def downscale(width, height):
    """
    Return the optimal percentage for downscaling
    :param width: width of the original image
    :param height: height of the original image
    :return: percentage for downscaling, in comparison to orig. picutre
    """
    # sidewards image
    if width > height:
        return (400 / width) * 100
    else:  # upwards image
        return (400 / height) * 100


class ImageMode:
    """Image Prototype Settings"""

    def __init__(self):
        # taken from the opencv repository on Github:
        # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        self.cascadeclass = "./Resources/Haarcascade/haarcascade_frontalface_default.xml"
        self.facecascade = None

    def updateCascadeClass(self, newcascadeclass):
        """
        TODO
        :param newcascadeclass:
        :return:
        """
        self.cascadeclass = "./Resources/Haarcascade/" + newcascadeclass
        print(self.cascadeclass)
        self.facecascade = cv2.CascadeClassifier(self.cascadeclass)  # for face recognition/detection

    def runImage(self, path, fer):
        """
        TODO
        :param path:
        :param fer:
        :return:
        """

        if self.facecascade is None:
            self.facecascade = cv2.CascadeClassifier(self.cascadeclass)  # for face recognition/detection

        frame = cv2.imread(path)

        # Downscaling of picture if too large, for better UserInterface Visualisation
        if frame.shape[0] > 400 or frame.shape[1] > 400:
            scale_percent = downscale(frame.shape[0], frame.shape[1])
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # Creating a grayscale image out of the RGB webcam frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect all Faces in the grayscale frame
        faces = (self.facecascade.detectMultiScale(
            gray,  # from grayscale image
            scaleFactor=1.8,
            minNeighbors=5,
            minSize=(22, 22),
        )
        )
        ROI = None  # variable holding found face
        # Place a emotion-label and a Rectangle around the found faces
        # x: Start Coordinate x in horizontal direction
        # y: Start Coordinate y in vertical direction
        # w: End Coordinate w in horizontal direction (width)
        # h: End Coordinate h in vertical direction (height)
        for (x, y, w, h) in faces:
            ROI = frame[y:y + h, x:x + w]  # store subimage/subface in ROI
            # Draw a rectangle around the face
            cv2.rectangle(frame,  # Desired Frame
                          (x, y),  # startpoint of frame
                          (x + w, y + h),  # endpoint of frame
                          (0, 255, 0),  # color of frame, set to green
                          2  # thickness of frame
                          )
            # Put Text (detected Emotion) to the found face
            emotion = fer.getEmotion(ROI)
            frame = cv2.putText(frame,
                                emotion,  # Message: Detected Emotion from trained Model
                                (x, y),  # Label position - face found on (x,y)
                                cv2.FONT_HERSHEY_SIMPLEX,  # Label Font
                                1,  # Font Scaling Factor
                                (255, 0, 0),  # Color of LabelText - set to blue
                                2,  # Thickness of Text in px
                                cv2.LINE_AA)  # LineType used

        return frame
