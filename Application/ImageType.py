"""CameraType.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Matr.-Nr.**: k11803108

**Description**: Bachelorthesis - Prototype for FER

In this File the Camera Prototype is running.
Taking a trained CNN for calssifing emotion of a choosen picture.
The picture has to be in the format of png,jpg or jpeg (watch out for ending of your file!)

Details on Usage:
TODO

Main Feautres of the Displaying and FaceTracking are done Using the OpenCV Library, which can be looked up
here: https://opencv.org/
"""
import tkinter
import tkinter.filedialog
import cv2
from datetime import datetime


class ImageType:
    """Image Prototype Settings"""
    def __init__(self):
        self.terminatekey = 3  # terminates the program with ctrl+c
        # taken from the opencv repository on Github:
        # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        self.cascadeclass = "./Resources/haarcascade_frontalface_default.xml"

    def runImage(self, imagepath):
        if imagepath == "":
            root = tkinter.Tk()
            path = tkinter.filedialog.askopenfilename()
            root.destroy()
        else:
            path = imagepath

        if not (path.endswith('.jpeg') or path.endswith('.jpg')or path.endswith('.png')):
            print("You did not select a Image with correct Format, only use jpeg,jpg or png!")
            return
        image = cv2.imread(path)
        cv2.imshow("FaceImage", image)
        while True:

            # check which key was pressed
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('c'):  # take a screenshot of current frame
                cv2.imwrite('./Screenshot/LatestCapture' + datetime.now().strftime("%m%d%Y,%H%M%S") + '.jpg', image)
            if pressedKey == self.terminatekey:  # close FER
                print("User Shutdown System")
                break  # breaks outer loop

            #  Release the Capture (Webcam)
        #  Close all Windows
        cv2.destroyAllWindows()
