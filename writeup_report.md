# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Nvidia_CNN.png "Model Visualization"
[image2]: ./examples/center-track.jpg "Center Track"
[image3]: ./examples/left_sample.jpg "Left Sample"
[image4]: ./examples/center_sample.jpg "Center Sample"
[image5]: ./examples/right_sample.jpg "Right Sample"
[image6]: ./examples/center-track.jpg "Normal Image"
[image7]: ./examples/center-flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolution neural network with 5x5 to 3x3 filter sizes and depths between 24 and 64 (model.py lines 103-107)

The model includes RELU layers to introduce nonlinearity (code line 103-114), and the data is normalized in the model using a Keras lambda layer (code line 101). Also, the model contain a Cropping2D layer to crop the region of interest (ROI) that excludes the sky/mountain on the top of images and the hood of the car on the bottom of images.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 109-115).

First, I randomly split the recording data sets into a train and validation set. Therefore, the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 84). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25), and the loss function used here is mean_squared_error to measure the average squared error between true steering angles and predicted steering angles in each batch.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and, in order to generalize the data set, I drive the car on reveser route to get more right-turn steering angle samples.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a small neural network to test how well it could train.

My first step was to use a convolution neural network model similar to the LeNet, and but It work not well with sharp curve (or I may not have enoough data at the begining). Then, I use the architecture from Nvidia's paper __"End-to-End Learning for Self-Driving Cars"__ I thought this model might be appropriate because the model is able to get the good result for train self-driving car in Nvidia's paper. However, the behavior I collected my drving data will also affect the learning result.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model (LeNet) had a continued decreasing mean squared error on the training set but the mean squared error on the validation set get increasing after certain layer (10-15). This implied that the model was overfitting.

To combat the overfitting, I modified the model with Nvidia's archetecture and added dropout layer (with 0.5 rate) after each full connectted layers so that the model will randomly dropout partial features to prevent overtrainning with all features.

Then I adjust the input arguments of 'adam' optimizer with the samll decay to reduce the learning rate after each epoch to avoid the possible of the divergence.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (such as some sharp turning area and the bridge with different texture of the ground) In order to improve the driving behavior in these cases, I record extra data when drving in those area.

Also, there is a trciky bug that I read images with `cv2.imread`, which construct it color channel as "BGR", but, in `drive.py`, it use `PIL.Image.open`, which construct its color channel as "RGB", to read images from the simulator. With this kind of bug, my model always predict the wrong angle when drving into the first shadow area in the second track. After, I swap the first and third channel of each image in `drive.py`, my model could predict the correct output in the shadow area of the second track. But still need more data pass sharp turning area of the second track.

At the end of the process, the vehicle is able to drive autonomously around the first track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 100-120) consisted of 5 convolution neural network and 4 fully connected layer with the following layers and layer sizes:

| Layer                 |     Description                                               |
|:---------------------:|:-------------------------------------------------------------:|
| Input                 | 160x320x3 Colorscale (BGR) image                              |
| Lambda layer          | Normalize image, outputs 160x320x3                            |
| Cropping2D            | outputs 85x320x3                                              |
| Convolution_1 5x5     | 1x1 stride, valid padding, outputs 81x316x24                  |
| RELU                  |                                                               |
| Max pooling           | 2x2 stride,  outputs 40x158x24                                |
| Convolution_2 5x5     | 1x1 stride, valid padding, outputs 36x154x36                  |
| RELU                  |                                                               |
| Max pooling           | 2x2 stride,  outputs 18x77x36                                 |
| Convolution_3 5x5     | 1x1 stride, valid padding, outputs 14x73x48                   |
| RELU                  |                                                               |
| Max pooling           | 2x2 stride,  outputs 7x36x48                                  |
| Convolution_4 3x3     | 1x1 stride, valid padding, outputs 5x34x64                    |
| RELU                  |                                                               |
| Convolution_5 3x3     | 1x1 stride, valid padding, outputs 3x32x64                    |
| RELU                  |                                                               |
| Fully connected 1     | Flatten previous layer, output 6144 features single layer     |
| Dropout layer         | dropping rate 50%                                             |
| Fully connected 2     | 6144 in 100 out                                               |
| RELU                  |                                                               |
| Dropout layer         | dropping rate 50%                                             |
| Fully connected 3     | 100 in 50 out                                                 |
| RELU                  |                                                               |
| Dropout layer         | dropping rate 50%                                             |
| Fully connected 4     | 50 in 10 out                                                  |
| RELU                  |                                                               |
| Dropout layer         | dropping rate 50%                                             |
| Output layer          | reduce to final 1 feature                                     |

Here is a visualization of the Nvidia architecture from their paper:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to control the 
steer when the car is more closed or far away from the lane edge. In order to use the images from left and right side, I also need to provide possible 
steering angle data for the left and right image. In the lecture, it is suggested to derive the steering angle from center images. For instance, if center image 
provide a left-turn angle, then the left image should provide a less left-turn angle. These images show what a recovery looks like starting from the left image to center image to right image:

| Left Camera           |     Center Camera     |    Right Camera       |
|:---------------------:|:---------------------:|:---------------------:|
| ![alt text][image3]   | ![alt text][image4]   |  ![alt text][image5]  |

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would collect more general target data (steering angle). Because we train the car 
to run the first track counterclockwise (CCW), we will get a left-turn bias data. Therefore, flipping images could unbias the dataset. For example, here is an 
image that has then been flipped:

|   Original Image    |     Flipped Image     |
|:-------------------:|:---------------------:|
| ![alt text][image6] | ![alt text][image7]   |

After the collection process, I had about 70,000 number of data points. I then preprocessed this data by normalizing image inside the model.
I finally randomly shuffled the data set and put 10% of the data into a validation set. I used this training data for training the model. The validation set 
helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the video "run1.mp4", which shows that 
the vehicle is able to drive autonomously around the first track without leaving the road. However, the vehicle can only drive 3/4 loop of the second track, 
I may need to collect more data to pass sharp turn under shadows. 
