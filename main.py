"""main.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Matr.-Nr.**: k11803108

**Description**: Bachelorthesis - Prototype for FER

In this main File ...

"""
from Application.CameraType import CameraType
from Application.ImageType import ImageType
import sys

"""Possible Argument Variables List"""
# Camera Mode
cameramode = "-cam"
internalcam = "-internal"
externalcam = "-external"

# Image Mode
imagemode = "-img"


def setDefault():
    prototype = 1  # 0: run expression detection with camera, 1: choose a image to be detected
    cameratype = internalcam  # user can set, if internal or external camera should be used
    imagepath = ""
    return prototype, cameratype, imagepath


# Method which checks the commandline arguments, if given
def checkCMDParameter():
    nrargs = len(sys.argv) - 1  # check number of found arguments
    cnt = 1  # first argument

    cameratype = ""
    imagepath = ""

    if nrargs != 0:  # check if user put on some arguments
        if sys.argv[cnt] == cameramode:  # check if user wants to use Realtime Mode
            prototype = 0
            cnt = cnt + 1
            if cnt <= nrargs:  # check if user also set cameratype (internal or external)
                if sys.argv[cnt] == internalcam or sys.argv[cnt] == externalcam:
                    cameratype = sys.argv[cnt]
                else:
                    prototype, cameratype, imagepath = setDefault()
                    print("You used the wrong Arguments by selecting Camera, continue with Default Settings")
            else:
                # Set Default Camera
                cameratype = internalcam
        elif sys.argv[cnt] == imagemode:  # check if user wants to use Image Mode
            prototype = 1
            cnt = cnt + 1
            if cnt <= nrargs:  # check if user also set path to image
                imagepath = sys.argv[cnt]
                print(imagepath)
        else:
            prototype, cameratype, imagepath = setDefault()
            print("You used the wrong Arguments - Mode does not exist, continue with Default Settings")
    else:
        print("No User Arguments given - Continue with Default Settings")
        prototype, cameratype, imagepath = setDefault()
    return prototype, cameratype, imagepath


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pro, cam, imgpath = checkCMDParameter()
    if pro == 0:
        camera = CameraType(cam)  # start webcam rendering
        camera.runCamera()
    else:
        image = ImageType()
        image.runImage(imgpath)
