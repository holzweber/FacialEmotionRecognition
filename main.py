"""main.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Matr.-Nr.**: k11803108

**Description**: Bachelorthesis - Prototype for FER

In this main File ...

Main Feautres of the Displaying and FaceTracking are done Using the OpenCV Library, which can be looked up
here: https://opencv.org/
"""
from Application.CameraType import CameraType
from Application.ImageType import ImageType

"""User Settings"""
prototype = 1  # 0: run expression detection with camera, 1: choose a image to be detected


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if prototype == 0:
        camera = CameraType()  # start webcam rendering
        camera.runCamera()
    else:
        image = ImageType()
        image.runImage()
