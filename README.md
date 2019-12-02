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

[image1]: ./examples/CNN_architecture.png "Model Visualization" 
[image2]: ./examples/model_keras.png "Grayscaling"
[image3]: ./examples/Training_the_neural_network.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode [I didn't make any change on this one]
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The requirement of getting the simulator to complete a full loop of Track 1 is not much. I was able to do so by implementing the NVIDIA's "[End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)." Precisely like the purpose of this project, I  focused on picking up a model and training the data rather than inventing a new one on my own. Below is the CNN Architecture of model_nvidia.    

![alt text][image1]

The following is the structure of the model.  

```sh
def model_nvidia(act='relu', d=0.5):

# based on Nvidia end-to-end driving model with customized activation and dropout for each layer (5 CNN + 3 FC)
    
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x/255-0.5, output_shape=(90, 320, 3)))
    model.add(Conv2D(24, (5,5), strides=(2,2), activation=act))
    model.add(Dropout(d))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation=act))
    model.add(Dropout(d))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation=act))
    model.add(Dropout(d))
    model.add(Conv2D(64, (3,3), activation=act))
    model.add(Dropout(d))
    model.add(Conv2D(64, (1,1), activation=act))
    model.add(Dropout(d))
    model.add(Flatten())
    model.add(Dense(100, activation=act))
    model.add(Dropout(d))
    model.add(Dense(50, activation=act))
    model.add(Dropout(d))
    model.add(Dense(10, activation=act))
    model.add(Dropout(d))
    model.add(Dense(1))
    return model
```

I used the Keras to implement this model, which consists of one Lambda layer (for grey-level normalization ), five convolutional layers ( for feature extraction), and three fully-connected layers (designed to function as a controller for steering) accordingly. Moreover, This model uses RELU as the activation function to introduce nonlinearity in each layer. Thus, I focus on finding the minimum threshold needed to let the simulator autonomously driving on Track 1 successfully, as I believe it is the robustness of the model. 

Below is an alternation of the model that I remove all dropout in the CNN layer because I read an article discussing whether to use dropout in CNN layers or not. So I would like to give it a shot.

```sh
def model_nvidia(act='relu', d=0.5):

#    based on nVidia end-to-end driving model with ustomized activation and dropout that only applied in the fully-connected layers 
    
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x/255-0.5, output_shape=(90, 320, 3)))
    model.add(Conv2D(24, (5,5), strides=(2,2), activation=act))
    #model.add(Dropout(d))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation=act))
    #model.add(Dropout(d))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation=act))
    #model.add(Dropout(d))
    model.add(Conv2D(64, (3,3), activation=act))
    #model.add(Dropout(d))
    model.add(Conv2D(64, (1,1), activation=act))
    #model.add(Dropout(d))
    model.add(Flatten())
    model.add(Dense(100, activation=act))
    model.add(Dropout(d))
    model.add(Dense(50, activation=act))
    model.add(Dropout(d))
    model.add(Dense(10, activation=act))
    #model.add(Dropout(d))
    model.add(Dense(1))
    return model
```
#### 2. Attempts to reduce overfitting in the model

I used the dropout function to avoid the overfitting. However, I noticed that most of my trials shown the validation loss seems smaller than the training loss. 

```sh
model = model_nvidia(act='relu', d=0.5)
model.compile(loss = 'mse', optimizer = 'adam')
```

Here is a table of my selected trials among many. With or without dropout in CNN layers, both can get a reasonable model to compete the Track 1 successfully.  

|         	|    model 1    	|    model 2    	|    model 3    	|    model 4    	|
|:-------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|
|  CNN(5) 	|   No dropout  	|   No dropout  	| dropout = 0.5 	| dropout = 0.5 	|
|   FC-3  	| dropout = 0.5 	| dropout = 0.4 	| dropout = 0.5 	|       -       	|
|   FC-2  	|       -       	|       -       	|       -       	| dropout = 0.5 	|
| outcome 	|       NG      	|       OK      	|       NG      	|       OK      	|


* Note: FC-2 means that the last dropout right after Dense(10, activation=act) is not used. 

To ensure the model was not overfitting, I set the train_samples and validation_samples by the following command: 

```sh
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

Models were tested by running them through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. As I mentioned preveiously, the only thing I tuned is the dropout value and where to insert the dropout layer (CNN or FC). 

```sh
model.compile(loss = 'mse', optimizer = 'adam')
```
#### 4. Appropriate training data

For this project, I first recorded two laps of Track 1 data and two laps of Track 2 data by keyboard control.  Later  I was focused on the model implementation. After I observed the simulation results,  I did retake data on purposely to control the speed at 9mph, which is the same speed of the vehicle in the simulator in the autonomous mode.  I used a combination of center-lane driving, tri-camera, and mirror images to train the data. For this demo, the submitted model was trained by Track 1 data only.   I tried to train a model for Track 2 but still no success, I can tell the difficulty in Track 2, but this is beyond my scope for now. I will get back later.  

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to pick up a published model rather than to invent a new one. My first step was to use a convolution neural network model from Nvidia, and I believe this model appropriate because it has been proven to be useful. To gauge how well the model was working, I split my image and steering angle data into a training and validation set.  I know the importance of the dropout layer from the previous project, so I didn't struggle with any overfitting. As I mentioned earlier,  most of the trials I had show that the validation loss is smaller than the training loss, and I am still in a process to understand it. 

The final step was to run the simulator to see how well the car was driving around track one. Because I used to dropout layer in the beginning,  the tricky part for me is that sometimes, the validation loss was so little (way below the training loss), but the model might not survive from running the whole track after all. Therefore, the MSE (mean squared error) may not be the best indicator in this case when we don't have any overfitting issue. 

There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, one can retake the data by driving slowly, staying in the center of the lane, and taking more laps; this is the minimum effort to do in Track one. For Track two,  shadow is one of issue to interfere with the training, so one can further implement a pre-image processing function to bring the mean brightness up in the generator function. I tried this, but it will slow down the training a lot. Therefore,  I decided to focus on completing the task of Track 1.  After all, the vehicle can drive autonomously around the track one without leaving the road with many of my different DOE models. 

#### 2. Final Model Architecture

The final model architecture consisted of one Lambda layer, five convolutional layers, and three fully-connected layers accordingly. I didn't try other models because this one is working well. Of course, I would like to make the Track 2 Challenge, then I will still use the same model, but add more pre-image process steps.  

I have described the model at the beginning of the session, and I will omit it here.

#### 3. Creation of the Training Set & Training Process


To capture good driving behavior, I first recorded two laps on Track 1 using center lane driving. 
Note: 
1. I did not record the vehicle recovering from the left side and right sides of the road back to the center, but the model learns to stay in the center pretty well. 

2. I did not use Track 2 data for the training. As I said, I want to know the minimum effort to make a model for completing the Track 1 loop. 

I randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or underfitting.
Most importantly, I on-purposely to bundle of those images from three cameras (left, center, and right) and their mirrored images altogether. It means at each timestamp, and there are always six images grouped when I do the random split into training and validation sets. I think this can potentially help to balance the learning of left- and right-steering.  

Note:  I set the epochs as 20, and batch size as 256. It works, so I didn't play these two parameters too much.