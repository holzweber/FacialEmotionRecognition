"""CameraMode.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Description**: Bachelorthesis - Prototype for FER

**Institution**: Johannes Kepler University Linz - Institute of Computational Perception

This file handles the general CameraMode. The user has already choosen an Camera Source in the Userinterface. If this
selected source is available, the user then will see a new Window with the current camera capture. Each frame will run
through a haarcascde, detecting all faces and see rectangles around each and everyone.
The all faces will run through a trained model, using a given ExpressionRecogniiton instance for emotion detection.
This label will then be placed to each and every detected face.

The new opened Frame can be closed using 'q'.
Using 'f' the current face will be stored in ./Screenshots/
Using 'c' the current frame (with FER) will be stored in ./Screenshots/

The default Haarcascade for facedetection is taken from the opencv repository on Github:
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

**Required installations for running this class**:
OpenCV: pip install opencv-python       https://pypi.org/project/opencv-python/
"""

import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
from datetime import datetime
import tkinter as tk
import tkinter.messagebox
from CNN.ExpressionRecognition import ExpressionRecognition


class CameraMode:
    """Camera Prototype Settings"""

    def __init__(self):

        self.cameraType = 0  # 0: internal, 1:external
        self.terminatekey = 'q'  # terminates the program with ctrl+c
        # taken from the opencv repository on Github:
        # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        self.cascadeclass = "./Resources/Haarcascade/haarcascade_frontalface_default.xml"
        self.facecascade = None

    def updateCascadeClass(self, newcascadeclass):
        """
        If a new facedetector should be used, this method will be called with the wanted .xml file
        :param newcascadeclass: File which will be used. Format has to be .xml
        :return:
        """
        self.cascadeclass = "./Resources/Haarcascade/" + newcascadeclass
        print(self.cascadeclass)
        self.facecascade = cv2.CascadeClassifier(self.cascadeclass)  # for face recognition/detection

    def runCamera(self, cam, fer):
        """
        This method starts the camera and a new window, where the user will see him/herself with the accorindg emotion
        Also multiface detection is possible.

        :param cam: camera to be used as string "Internal" or "External"
        :param fer: FER instance to be used
        :return:  statistic vector in % for imaging
        """
        if cam == "Internal":
            print("Selected Internal Camera")
            self.cameratype = 0  # 0: internalwebcam, 1: externalwebcam
        elif cam == "External":
            print("Selected External Camera")
            self.cameratype = 1  # 0: internalwebcam, 1: externalwebcam
        else:
            print("ERROR: No camera selected")
            return

        self.facecascade = cv2.CascadeClassifier(self.cascadeclass)  # for face recognition/detection

        cap = cv2.VideoCapture(self.cameratype, cv2.CAP_DSHOW)  # opens up the capture channel of the webcam
        if cap is None or not cap.isOpened():
            tk.messagebox.showerror(title="Camera Error", message="Camera not available!")
            print("ERROR: Selected Camera is not available")
            return

        emotionCounter = [0, 0, 0, 0, 0, 0, 0]

        # Create window
        # kick off the GUI
        windowtitle = "Holzweber_11803108_FER - press c to take screenshot and f to store face"
        ret, frame = cap.read()
        cv2.imshow(windowtitle, frame)

        # while window is not closed
        while cv2.getWindowProperty(windowtitle, cv2.WND_PROP_VISIBLE) > 0:  # while window not closed
            # Capture frame-by-frame - reading in current frame
            ret, frame = cap.read()

            # Creating a grayscale image out of the RGB webcam frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect all Faces in the grayscale frame
            faces = (self.facecascade.detectMultiScale(
                gray,  # from grayscale image
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
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
                emotion, predictions = fer.getEmotion(ROI)
                frame = cv2.putText(frame,
                                    emotion,  # Message: Detected Emotion from trained Model
                                    (x, y),  # Label position - face found on (x,y)
                                    cv2.FONT_HERSHEY_SIMPLEX,  # Label Font
                                    1,  # Font Scaling Factor
                                    (255, 0, 0),  # Color of LabelText - set to blue
                                    2,  # Thickness of Text in px
                                    cv2.LINE_AA)  # LineType used
                emotionCounter[np.argmax(predictions)] += 1
            cv2.imshow(windowtitle, frame)

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
        return (emotionCounter / np.sum(emotionCounter)) * 100  # return statistic vektor in percent
