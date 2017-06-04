**Behavioral Cloning**

Writeup Template

You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/1.png "Left, Center, Right image"
[image2]: ./examples/2.png "Recovery"
[image3]: ./examples/3.png "Recovery Image"
[image4]: ./examples/4.png "Flipped Images"
[image5]: ./examples/5.png "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 for Final project recording

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The Trial-color.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Firstly, started with a few layers
1st - Convolutional (5,3,3), RELU
Flatten()
fully connected - Dense(100)
fully connected - Dense(1)

but the model couldn't get fit well with the training dataset during the processing and simulator running
Started adding layers one by one.
Too prevent overfitting added dropout each layers
For better speed, I added maxpooling each layers
(I used amazon AWS at the beginning. then, I paid over $100 for a month. then I decided not to use it even though I could get a result faster if I use it.)

Convolutional neural network model - Regression

Batch size = 64
No. of epoch = 100
Learning rate = 0.001
Model optimizer = adam
Keras callback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 5)

Input Image (60, 120, 3)
cropped image (30,120,1) - Top 20, Bottom 10.

1st - Convolutional (5,3,3), RELU, Maxpooling(2x2), Dropout(0.5)
2nd - Convolutional (5,3,3), RELU, Maxpooling(2x2), Dropout(0.5)
3rd - Convolutional (24,5,5), RELU, Maxpooling(2x2), Dropout(0.5)
4th - Convolutional (36,5,5), RELU, Maxpooling(2x2), Dropout(0.5)
5th - Convolutional (64,3,3), RELU, Dropout(0.5)

Flatten
fully connected - Dense(1000), Dropout(0.5)
fully connected - Dense(100), Dropout(0.5)
fully connected - Dense(1)

####2. Attempts to reduce overfitting in the model

The model contains dropout, Maxpooling each layers in order to reduce overfitting (Trial-Color.ipynb model architecture). The model also contains flipped images for the generalized images

The model was trained and validated on different data sets to ensure that the model was not overfitting (Model architecture- train test split). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was tuned manually, 0.001

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center, left, right lane driving, recovering from the left and right sides of the road

Normal center driving
drivingLog1
drivingLog2
drivingLog3
drivingLog4
drivingLog5

Recover driving (Left to center, right to center)
drivingLog6

Opposite driving
drivingLog7

Finally, drivingLog1, drivingLog2, drivingLog3, drivingLog4, drivingLog5, drivingLog6 were used during the preprocessing.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to collect large amounts of data set as well as manipulated data set by flipping images for the generalization

Firstly, collected dataset with joystick trials separately and then merged with selected dataset (drivingLog3,4,5,6 and some of drivingLog1). To prevent overfitting, I regenerated flipped images. but if the steering angle is zero I didn't flipped it because I already got too many zero steering angles. it could prevent processing large data set which could make process slow. Also, left and right images were used by adding steering angle +0.25 for left images and -0.25 for right images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I checked the training loss and val_loss whether the value is low for both train data and validation data.

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (Trial-Color.ipynb Model architecture) consisted of a convolution neural network with the following layers and layer sizes

Convolutional neural network model - Regression

Batch size = 64
No. of epoch = 100
Learning rate = 0.001
Model optimizer = adam
Keras callback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 5)

Input Image (60, 120, 3)
cropped image (30,120,1) - Top 20, Bottom 10.

1st - Convolutional (5,3,3), RELU, Maxpooling(2x2), Dropout(0.5)
2nd - Convolutional (5,3,3), RELU, Maxpooling(2x2), Dropout(0.5)
3rd - Convolutional (24,5,5), RELU, Maxpooling(2x2), Dropout(0.5)
4th - Convolutional (36,5,5), RELU, Maxpooling(2x2), Dropout(0.5)
5th - Convolutional (64,3,3), RELU, Dropout(0.5)

Flatten
fully connected - Dense(1000), Dropout(0.5)
fully connected - Dense(100), Dropout(0.5)
fully connected - Dense(1)


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded multiple laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to center line. These images show what a recovery looks like starting from the side road :

![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would reduce the overfitting For example, here is an image that has then been flipped:

![alt text][image4]

After the collection process and image preprocessing, I had around 60,000 number of data points. Then, I threw away some of images which had a lot of similar images in a specific angles, such as 0, 0.25, -0.25. Finally, I could get around 30,000 images.

I finally shuffled the data set and split 70% training data and 30% validation data

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
