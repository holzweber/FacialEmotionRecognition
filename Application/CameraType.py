"""CameraType.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Matr.-Nr.**: k11803108

**Description**: Bachelorthesis - Prototype for FER

In this File the Camera Mode Prototype is running.
Taking a trained CNN for classification of emotion of a videocaptured picture (frame). Therefore it is possible
to use the FER in a Realtime simulation.

Main Feautres of the Displaying and FaceTracking are done Using the OpenCV Library, which can be looked up
here: https://opencv.org/
"""

import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
from datetime import datetime

from CNN.ExpressionRecognition import ExpressionRecognition


class CameraType:
    """Camera Prototype Settings"""
    def __init__(self, cameratype):
        if cameratype == "-internal":
            print("Selected Internal Camera")
            self.cameratype = 0  # 0: internalwebcam, 1: externalwebcam
        else:
            print("Selected External Camera")
            self.cameratype = 1  # 0: internalwebcam, 1: externalwebcam
        self.terminatekey = 'q'  # terminates the program with ctrl+c
        # taken from the opencv repository on Github:
        # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        self.cascadeclass = "./Resources/haarcascade_frontalface_default.xml"

    def runCamera(self):
        fer = ExpressionRecognition() # create class for recognition

        facecascade = cv2.CascadeClassifier(self.cascadeclass)  # for face recognition/detection

        cap = cv2.VideoCapture(self.cameratype, cv2.CAP_DSHOW)  # opens up the capture channel of the webcam
        # Create window
        # cv2.namedWindow("Holzweber_11803108_FER")
        # cv2.createButton("Back", back, None, cv2.QT_PUSH_BUTTON, 1)
        # kick off the GUI

        while True:
            # Capture frame-by-frame - reading in current frame
            ret, frame = cap.read()

            # Creating a grayscale image out of the RGB webcam frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect all Faces in the grayscale frame
            faces = (facecascade.detectMultiScale(
                gray,  # from grayscale image
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100),
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

            cv2.imshow("Holzweber_11803108_FER", frame)

            # check which key was pressed
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('c'):  # take a screenshot of current frame
                cv2.imwrite('./Screenshot/LatestCapture' + datetime.now().strftime("%m%d%Y,%H%M%S") + '.jpg', frame)
            if pressedKey == ord('f'):  # take a screenshot of current face
                if ROI is not None:
                    print("Took Screenshot of Detected Face in GrayScale Format")
                    cv2.imwrite('./Screenshot/LatestFace' + datetime.now().strftime("%m%d%Y,%H%M%S") + '.jpg', ROI)
            if pressedKey == ord(self.terminatekey):  # close FER
                break  # breaks outer loop

        #  Release the Capture (Webcam)
        cap.release()
        #  Close all Windows
        cv2.destroyAllWindows()
