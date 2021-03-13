"""ImageMode.py

# Facial Expression Recognition
**Author**: Christopher Holzweber

**Description**: Bachelorthesis - Prototype for FER

**Institution**: Johannes Kepler University Linz - Institute of Computational Perception

This file handles the general ImageMode. The user has already selected an image from a local directory in the
UserInterface.py class. Using the path to the image, this class will first check, if the image needs to be downscaled,
and then does the FER using the selected Haarcascade and instance of the ExpressionRecogniton class.

The default Haarcascade for facedetection is taken from the opencv repository on Github:
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

**Required installations for running this class**:
OpenCV: pip install opencv-python       https://pypi.org/project/opencv-python/
"""

import cv2  # OpenCV for imagehandling


def downscale(width, height,threshold):
    """
    Return the optimal percentage for downscaling
    :param width: width of the original image
    :param height: height of the original image
    :param threshold: Threshold for downscaling.
    :return: percentage for downscaling, in comparison to orig. picutre
    """
    # sidewards image
    if width > height:
        return threshold / width
    else:  # upwards image
        return threshold / height


class ImageMode:
    """ImageMode class handles the FER of a static image, given by the user"""

    def __init__(self):
        # taken from the opencv repository on Github:
        # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        self.cascadeclass = "./Resources/Haarcascade/haarcascade_frontalface_default.xml"
        self.facecascade = None
        self.downscaleThreshold = 500

    def updateCascadeClass(self, newcascadeclass):
        """
        If a new facedetector should be used, this method will be called with the wanted .xml file
        :param newcascadeclass: File which will be used. Format has to be .xml
        :return:
        """
        self.cascadeclass = "./Resources/Haarcascade/" + newcascadeclass
        self.facecascade = cv2.CascadeClassifier(self.cascadeclass)  # for face recognition/detection

    def runImage(self, path, fer):
        """
        This method takes an image path of a local directory and also a Expression Recognition instance to be used
        Then using OpenCV all faces in the images will be detected (shown with squares).
        Then using the FER an emotion label will be added to each square. The new image will be returned to the caller
        added
        :param path: Path of the oringinal image from a local directory with format jpeg,JPG,jpg,png or PNG
        :param fer: ExpressionRecognition instance to be used for detecting emotion
        :return: FER Image
        """

        if self.facecascade is None:
            self.facecascade = cv2.CascadeClassifier(self.cascadeclass)  # for face recognition/detection

        frame = cv2.imread(path)  # get image from local directory

        # Downscaling of picture if too large, for better UserInterface Visualisation
        if frame.shape[0] > self.downscaleThreshold or frame.shape[1] > self.downscaleThreshold:
            scale_percent = downscale(frame.shape[0], frame.shape[1], self.downscaleThreshold)
            width = int(frame.shape[1] * scale_percent)
            height = int(frame.shape[0] * scale_percent)
            dim = (width, height)
            # resize image
            frame = cv2.resize(frame, dim)

        # Creating a grayscale image out of the RGB webcam frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all Faces in the grayscale frame
        faces = (self.facecascade.detectMultiScale(
            gray,  # from grayscale image
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(38, 38),
        )
        )

        ROI = None  # variable holding found face
        # Place a emotion-label and a Rectangle around the found faces
        # x: Start Coordinate x in horizontal direction
        # y: Start Coordinate y in vertical direction
        # w: End Coordinate w in horizontal direction (width)
        # h: End Coordinate h in vertical direction (height)
        for (x, y, w, h) in faces:
            rawface = frame[y:y + h, x:x + w]  # store subimage/subface in ROI
            # Draw a rectangle around the face
            cv2.rectangle(frame,  # Desired Frame
                          (x, y),  # startpoint of frame
                          (x + w, y + h),  # endpoint of frame
                          (0, 255, 0),  # color of frame, set to green
                          2  # thickness of frame
                          )
            # Put Text (detected Emotion) to the found face
            emotion, predictions = fer.getEmotion(rawface)
            frame = cv2.putText(frame,
                                emotion,  # Message: Detected Emotion from trained Model
                                (x, y),  # Label position - face found on (x,y)
                                cv2.FONT_HERSHEY_SIMPLEX,  # Label Font
                                1,  # Font Scaling Factor
                                (255, 0, 0),  # Color of LabelText - set to blue
                                2,  # Thickness of Text in px
                                cv2.LINE_AA)  # LineType used

        return frame
