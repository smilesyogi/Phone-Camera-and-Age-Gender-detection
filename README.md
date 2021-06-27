# Phone-Camera-and-Age-Gender-detection
PHONE CAMERA AND AGE GENDER DETECTION IS A MACHINE LEARNIND AND DATA SCIENCE BASED Project Have you ever being in a situation to guess another person's age? Maybe Yes..! So here we are representing a very interesting project based on machine learning and data Science …it’s also like a fun activity while executing and definitely will make you curious to see the results..! So let’s go for it..!!! we are taking face , age and gender model for traing our machine for our program and taken those files into our main program file sdl.py file for implemantion. In this program we are capturing the face in rectangel of 227*227 and then taking that box into face model and performing computation on that as data to the model and giving predicted out put on screen.

Project Report On 
“Phone Camera and Age Gender Detection With Python”


SUBMITTED TO SAVITRIBAI PHULE PUNE UNIVERSITY, PUNE BACHELOR OF ENGINEERING (Computer Engineering) By 


1.Yogesh Chandewar (TCOB78) 
2.Ritesh Patil (TCOB76)
3.Jay Bayas (TCOB46)  
4.Nirmala Bhairamadgi (TCOB48) 




UNDER THE GUIDANCE OF Prof. Laxman Khandare 

 
Department of Computer Engineering Dr.D.Y.Patil Institute of Technology Pimpri, Pune-18 2020-2021 

 
Dr.D.Y.Patil Institute of Technology Pimpri, Pune-18 
Department of Computer Engineering 
CERTIFICATE 
This is to certify that the project entitled 
“Phone Camera and Age Gender Detection With Python”
Submitted By 

->Yogesh Chandewar (TCOB78) 
->Ritesh Patil (TCOB76) 
->Jay Bayas (TCOB46) 
->Nirmala Bhairamadgi (TCOB48) 




is a bonafide work carried out by them under the supervision of Prof. Laxman Khandare sir and it is approved for the partial fulfillment of the requirement of Savitribai Phule Pune University, Pune for the award of the Degree of Bachelor of Engineering(Computer Engineering )
. 

Prof. Laxman Khandare                                                                Dr. Santosh V. Chobe
    Internal Guide                                                                                               H.O.D 
Dept. of Computer Engg.                                                                     Dept. of Computer Engg. 

Dr. Pramod. D. Patil 
Principal 
Dr. D. Y. Patil Institute of Technology, Pimpri, Pune 



Signature of Internal Examiner  



                                             Signature of External Examiner  










Content 
Sr.No 	Name 	Page No 
1 	Introduction 	6
2 	Objective 	7
3 	Theory 	9
4 	Requirement Analysis 	12
5 	System Design 	13
6 	Implementation 	17
7 	System Requirement 	19
8 	Testing  	20
9 	Advantages 	21
10 	Future Scope 	24
11 	Conclusion 	27
12	OUTPUT	28
13 	References 	32




List of Figure 
Sr.No 	Name 	Page No 
01	Methodology	09
02	Parameter of LBM & LCP Method	09
03	Enhanced Histogram	10
04	Recognition	11
05	System Requirement	13




List of Table 
Sr.No 	Name 	Page No 
1 	Software requirement  	19
2 	Hardware requirement 	22






















                  Introduction 
PHONE CAMERA AND AGE GENDER DETECTION IS A MACHINE LEARNIND AND DATA SCIENCE BASED Project 
Have you ever being in a situation to guess another person's age? Maybe Yes..! 

So here we are representing a very interesting project based on machine learning and data Science …it’s also like a fun activity while executing and definitely will make you curious to see the results..!

So let’s go for it..!!!
we are taking face , age and gender model for traing our machine for our program and taken those files into our main program file sdl.py file for implemantion. 

In this program we are capturing the face in rectangel of 227*227 and then taking that box into face model and performing computation on that as data to the model and giving predicted out put on screen.
To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.






                   Objectives
1)Face Detection:
Import the required libraries 
Load the pre-trained haar cascade classifier for face detection from the storage.
2)To build a gender and age :
Detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.



3)Motion Detection
The last article covered live motion detection using OpenCV with the help of a web camera but it has several limitations as discussed above because it will bound to use at only one place. Instead, use your Android camera to keep track of changes at any place concerning a particular frame at a particular time. 
Let’s suppose you are doubting that someone behind you makes changes in your workplace but you can’t be able to figure it out as CCTVs are too costlier to afford but this approach is more economical and easier to keep track of of victim.









                       Theory 
we are taking face , age and gender model for traing our machine for our program and taken those files into our main program file detect.py file for implemantion. 
In this program we are capturing the face in rectangel of 
227*227 and then taking that box into face model and performing computation on that as data to the model and giving predicted out put on screen.
FOR ITS IMPLEMENTATION WE ARE USING FOLLOWING THINGS:-
1.ipweb cam:
An Internet Protocol camera, or IP camera, is a type of digital video camera that receives control data and sends image data via an IP network. ... They are commonly used for surveillance but unlike analog closed-circuit television (CCTV) cameras, they require no local recording device, only a local area network.

2.opencv(vedio capture , blob function ):
OpenCV ( Open Source Computer Vision Library) is an open source software library for computer vision and machine learning. OpenCV was created to provide a shared infrastructure for applications for computer vision and to speed up the use of machine perception in consumer products. OpenCV, as a BSD-licensed software, makes it simple for companies to use and change the code. There are some predefined packages and libraries that make our life simple and OpenCV is one of them

3.argparsesr:
The argparse module makes it easy to write user-friendly command-line interfaces. It parses the defined arguments from the sys. argv . The argparse module also automatically generates help and usage messages, and issues errors when users give the program invalid arguments.

4.math lib:
The Python Math Library provides us access to some common math functions and constants in Python, which we can use throughout our code for more complex mathematical computations. The library is a built-in Python module, therefore you don't have to do any installation to use it.


Using a phone camera with Python is very useful for those who are planning to create computer vision apps that will use a smartphone camera as a part of your application. Here I am using Python on Windows 10. Hope this works for other operating systems as well, but if you are using Windows then don’t worry just follow the steps mentioned below.
            

The process of using a phone camera with Python:

First, install the OpenCV library in Python; pip install opencv-python.
Download and install the IP Webcam application on your smartphones.
After installing the IP Webcam app, make sure your phone and PC are connected to the same network. Run the app on your phone and click Start Server.
After that, your camera will open with an IP address at the bottom. Copy the IP address as we will need to use it in our Python code to open your phone’s camera.













	Requirement Analysis 
Age detection is the process of automatically discerning the ageofapersonsolely from a photo of their face. Typically, you’ll see age detection implemented as a two-stage process:
1. Stage #1:Detect faces in the input image/video stream
2. Stage #2:Extract the face Region of Interest (ROI), and apply the age detector algorithm to predict the
age of the person
For Stage #1, any face detector capable of producing bounding boxes for faces in an image can be used
Once your face detector has produced the bounding box coordinates of the face in the image/video stream, you can move on to Stage #2 — identifying the age of the person.
MODULES
• CV2: OpenCV is a high performance library for digital image processing and computer vision, which is free and open source.
• X KERAS : Keras is an open-source high-level neural network API, written in python. It allows easy and fast prototyping.
• TENSORFLOW : Tensorflow is a free and open- source software library for dataflow and differentiable programming across a range of tasks. It is used in neural networks.

	                    System Design  
 

 


	 

	Implementation 
Download and install IP Webcam application on your mobile phone.
Then make sure your PC and Phone both are connected to the same network. Open your IP Webcam application on your both, click “Start Server” (usually found at the bottom). This will open a camera on your Phone.
A URL is being displayed on the Phone screen, type the same URL on your PC browser, and under “Video renderer” Section, click on “Javascript”.You can see video captured on your phone, which starts showing up on your browser. Now, what we will be going to do is, taking image data from the URL using the request module and convert this to an image frame using NumPy, and finally, start using our Android camera as a webcam in Python.
In the code:
Import module
Add URL displayed in your phone
Continuous fetch data from URL
Keep displaying this data collected
Close window
And now for images and video data as input through pc web cam we need not to give arguments in cmd.
And for image ,video and url as input data we need to give command line argument under --image <file_name>
And this argumet (file_name) is using argparser module for its implementation 
 









		System Requirement 


 

An IP camera can be accessed in opencv by providing the streaming URL of the camera in the constructor of cv2.VideoCapture.
Usually, RTSP or HTTP protocol is used by the camera to stream video.

WINDOWS 10, 
PYTNON 3.9.1,

OPENCV VERSION 4.5.2,
ARGPARSHER 
MATH

Testing 
FOR TESTING PURPOSE WE HAVE TO TEST FOR 1.PHONE CAMERA, 2.PC WEBCAM, 3. SAVED IMGES, 4.SAVED VIDEO, 5. IMAGE URL  
1. PHONE CAMERA: FOR THIS WE TOOK LIVE FEED OF PHONE AS INPUT TO OUT PROGARM BY THE IP GENERATED BY IP WEBCAM ANDROID APP
AND DONE THE TESTING
2.PC WEBCAM : FOR THE PC WEB CAM WE TOOK PC LIVE FEED AS INPUT AND DONE TESTING 
3.SAVED IMAGES:
FOR SAVED IMAGES WE TOOK IMAGE NAME AS INPUT TO OUR PROGRAM AND DONE TESTING ND FOR THIS TESTING TO BE DONE THE IMAGE NEED TO BE SAVED IN THE SAME FOLDER WHERE THE PROGRAM FILE IS SAVED 
4 .SAVED VIDEO: FOR SAVED VIDEO WE TOOK VIDEO NAME AS INPUT TO OUR PROGRAM AND DONE TESTING AND FOR THIS TESTING TO BE DONE THE VIDEO NEED TO BE SAVED IN THE SAME FOLDER WHERE THE PROGRAM FILE IS SAVED
videos are converted into image frames while implementation 
  5.IMAGE URL: for this testing we need to take any url form internet and paste this as argument to -image < url >
 

	
	Advantages 

1. Event Detection:
CCTV cameras are everywhere around us in offices, roads, hospitals, banks, train stations, parking lots, etc and 24/7 surveillance is difficult. The computer vision techniques allow us to monitor the events in real-time and detect anomalies or any specific action detection. Automatic Number Plate Recognition (ANPR) system can be used to control automatic gates, vehicle tracking, analyzing crowd and counting number of people.
2. Industrial Automation:
Automatic inspection of objects and classifying them into different categories play a major role in manufacturing industries. Industrial robots use computer vision algorithms to perform various tasks like separating two or more different objects into their respective categories, detecting whether the product is labeled or not. If not, then we can reject the product on the conveyor belt.

3. Medical Image Processing:
The progress in computer vision has led to extensive use of the medical imaging data to provide us with better prediction, diagnosis, and treatment of diseases. Some examples of this are the detection of tumors, arteriosclerosis or other malign changes, measurement of organ dimensions, blood flow, enhancement of ultrasonic or X-ray images that are interpreted by humans.
4. Self-driving Vehicles:
Artificial Intelligence is on the boom right now and companies are investing in self-driving technology. Computer vision is used to detect lanes and find a path for autonomous vehicles. Information from various sensors is analyzed and used to detect objects on the path like traffic lights, traffic signs and according to obstacles, we decide the appropriate action that needs to be taken.
5. Military Applications:
The military is probably one of the largest areas for computer vision. More advanced systems for missile guidance are developed to make dynamic decisions based on the information provided by various sensors including image sensors. Modern military concepts are emerging like “Battlefield awareness” in which strategic decisions are made after analyzing the information provided by sensors, detecting enemy soldiers or vehicles.	 












	Future Scope 
The model proposed was developed very carefully and error-free while being efficient. During this research, we proposed a model to estimate people's age by feeding the CNN image dataset, a deep learning algorithm and trained in broad database face-recognition. In all, we think that the accuracy of the model is decent and better than many already existing model, but can be further improved by using more data , data increase and better network architecture. The project model also predicts the age of the image provided with little slip and angle issue.The completely automated face recognition program was not sufficiently reliable to achieve high accuracy of recognition. It was mainly due to the fact that even a slight invariance to the size, rotation or shift errors of the segmented facial image did not occur in the face recognizing subsystem.
This project allows us to obtain useful knowledge about a variety of topics such as deep learning, the use of different libraries such as Keras, Pil, Seaborn, Tensorflow. The entire model is protected and this project has also enabled us to understand the stages of a project's creation and the working together. We have also learned how to test various project features.This project has given us great pleasure in creating a concept that can be used for good purposes and health in real life. In our project, there is ample scope for further development. For addition, a variety of features such as gender and age can be applied to this program. Outside of classification for age prediction, a regression model may also be used, if enough data is available. Through developing this project further, camera footage for safety purposes can be used for real-time age prediction.Nevertheless, if a more processing such as an eye detection technology has been applied to further normalize the segmented facial image, the output will expand to levels comparable to the manual facial detection and recognition system. This is one of the system needs identified in this section. Good techniques such as iris or retina recognition and facial recognition are used for user access and user authentication applications using the thermal range as this requires very high precision. The automatic real time system will be perfect for crowd control application.
Invariant face detection and recognition systems.In order to be used in simple surveillance applications, such as ATM user security, the fully automated face detection and recognition System (with an eyeshoot detection system) could be applied, whereas manual face detection and an automated recognition system is ideal for the mug shot matching. Implementation of a technique for eye detection would be a small extension to the system implemented and require little additional investigation. All other methods have shown good results and rely on the deformable prototype and main component analysis strategies.

		














                                             Conclusion 
As we have seen in this project that in just a few lines of code we have built an age and gender detection model, from here on you can also incorporate emotion detection and object detection in the same model and create a fully functional application.
Overall, We think the accuracy of the models is decent but can be improved further by using more data, data augmentation and better network architectures.
Hopefully, you found this project to be a good  and useful in your quest for recognizing a person’s age and gender. 















                  Output


                               



	 







             Reference 

1. 
https://data-flair.training/blogs/python-project-gender-age-detection/
2. https://www.sciencedirect.com/science/article/abs/pii/S0031320317302546
3. https://docs.openvinotoolkit.org/latest/omz_models_model_age_gender_recognition_retail_0013.html
4.
 https://www.geeksforgeeks.org/connect-your-android-phone-camera-to-opencv-python/
5.
 https://medium.com/@jeppbautista/connect-android-camera-to-python-using-opencv-90fd19d838
