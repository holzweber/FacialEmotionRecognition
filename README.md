# Facial Emotion Recognition (FER)
This is a Python based Prototype of a Facial Expression Recognition Tool. Using Tensorflow, Keras and OpenCV as Basic Frameworks for ML/DL and Visualisation.

# Motivation
This protoype is used as Bachelorthesis Project in my current Computer Science Study at the Johannes Kepler University Linz - Institute of Computational Perception.
The main motivation was to get used to common ML frameworks such as Tensorflow and using gained experience from basic ML curses at the university.

FER is therefore a great task to solve, since this is a very important topic in terms of Automated Feedbackgeneration, Human Robot Interaction and many more areas.

# What can you expect from this prototype?

Since this is only a prototype, one should not expect to find trained models with 100 accuracy. Indeed the given models score much lower. Eventough overfitting and training with small datasets can be nerve taking, playing around with the provided userinterface in camera or imagemode, will make a lot of fun, since the prototype is performing very will, eventough statistic values are not over the top.

Using good documentation you can use this prototype as basis for your own FER tool, by just train your own models and putting it to the according directories. 
# Uploaded Models 
In the directory Resources/Models you can find the best trained models. In the directory ML you can find the jupyter notebooks for training the models on your own.


model_vggface_fer2013_p224_dim3:

![model_vggface_fer2013_p224_dim3](https://user-images.githubusercontent.com/48522299/111029773-bebe3380-83fe-11eb-86a7-b4941eb3291c.png)


model_vgg_fer2013_p48_dim1:

![model_vgg_fer2013_p48_dim1](https://user-images.githubusercontent.com/48522299/111029121-30947e00-83fb-11eb-8528-8695205eba73.png)

model2020_FERG256_100px:

![model2020_FERG256_100px](https://user-images.githubusercontent.com/48522299/111029553-88cc7f80-83fd-11eb-8546-ca5002f7038c.png)

Attention: Even tough the FERG data performs optimal on its training data - for real human faces it performs pretty bad!

Thats why the FERG256 model was used on the entire fer2013 dataset (seen as testing data with normalization):

![model2020_FERG256_100px_ON_FER2013](https://user-images.githubusercontent.com/48522299/111029685-46f00900-83fe-11eb-9536-c475f00376f7.png)


model2020_fer2013_p48_dim1:

![model2020_fer2013_p48_dim1](https://user-images.githubusercontent.com/48522299/111029136-4144f400-83fb-11eb-84f5-81c916e0cabe.png)


# Links to used Frameworks 
Since this is the first time I was dealing with a ML task and even a Python project, i would like to mention the frameworks and documentations, that helped me a lot putting all this together:

https://www.tensorflow.org/ - main ML framework used, it also implies keras:

https://keras.io/ - building up CNN architectures

https://opencv.org/ - Mainly used for imageprocessing and facedetection using haarcascades.

Important: Watch out which version is installed, since some versions of OpenCV will produce memory leaks (which results in non-terminating programms)
As reported by me in the OpenCV forum, also this memory leak happens in the latest versio. 4.5.1.48


In order to run without increasing memory loss I switched back using opencv-contrib-python==4.1.2.30

https://colab.research.google.com/ - For training CNN Models on small datasets in a small amount of times. 

If you are not willing to use a cloud software for training your models, I also can mention:

https://jupyter.org/

For creating a small userinterface I will also mention

https://pypi.org/project/pygubu/ for creating a GUI using a desinger and

https://docs.python.org/3/library/tkinter.html?highlight=tkinter#module-tkinter
# Links to used Databases

FER2013 Database: https://www.kaggle.com/msambare/fer2013

FERG-DB: http://grail.cs.washington.edu/projects/deepexpr/ferg-2d-db.html

# Open Topics
I want to bring on more thoughts, how this prototype can be improved:
Using not only image data, but also e.g. voice snippets from the microphone, one can predict emotions much better.
Using bigger databases, one can train more satisfying models.
There exist more strategies on datapreprocessing like, attentionbased CNNs or Transferlearning, hybrid models using CNN and RNNs.

# Required Installations for using my protoype
I will list the installations, using the python installer (using Google Collab for training does not require them, since it is already done for you there):

Pygubu: https://pypi.org/project/pygubu/

        pip install pygubu              

        pip install pygubu-designer
        
OpenCV: https://pypi.org/project/opencv-python/

        pip install opencv-python       

Tensorflow: https://www.tensorflow.org/

        pip install tensorflow     

# Setup
The protoype was created using PyCharm running on Windows 10.

Prozessor:	AMD Ryzen 7 4700U with Radeon Graphics  2.00 GHz

Installed RAM:	16,0 GB

Systemtyp	64-Bit-OS, x64-based Prozessor

The training was done both in Jupyter Notebook and Google Colab.

# Acknowledgment
I want to thank my supervisor a.Univ.-Prof, Dr. Josef Scharinger for making this project possible.
