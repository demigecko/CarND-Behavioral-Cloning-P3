# note: my local desktop is Windows 10, so the format of the csv directory path is different from Mac 

# set the parameters 
correction =0.2; 
validation_split = 0.2
batch_size = 256

# import the packages
import os, platform, glob, csv, cv2
import numpy as np
import random
import sklearn
import math
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, AveragePooling2D, Activation, MaxPooling2D, BatchNormalization 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline

# I combined all the drive.log files into one manually 
csv_path = 'C:\\Users\\hsiny\\GitHub\\data\\Ho_track2\\driving_log.csv'
# I collected all the images in the same folder
IMG_path = 'C:\\Users\\hsiny\\GitHub\\data\\Ho_track2\\IMG\\'

# the model based on nVidia end-to-end driving model with ustomized activation and dropout
def model_nvidia(act='relu', d=0.5): 
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

# the generator 
def generator(samples, batch_size=256):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for j in range(3):
                    factor=[0, 1, -1]    
                    #filename = IMG_path + batch_sample[j].split('/')[-1]  #Unix or Mac
                    filename = batch_sample[j].split('/')[-1] # window 
                    # includes left, center, and right images with corresponding steering angle correction
                    measurement = round(float(batch_sample[3]) + factor[j]*correction,3); 
                    image = cv2.imread(filename)
                    images.append(image)
                    angles.append(measurement)
                    # produce the mirror images 
                    images.append(cv2.flip(image,1))
                    angles.append(measurement*(-1))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# data input                         
samples = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for line in reader:
        samples.append(line)
        
# set the train samples (0.8) and validation samples (0.2)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# calculate the numbers of steps and stpes_per_epoch
number_valid_steps = math.ceil(len(validation_samples*3*2)/batch_size)
steps_per_epoch = math.ceil(len(train_samples*3*2)/batch_size)

# run the model 
model = model_nvidia(act='relu', d=0.5)
model.compile(loss = 'mse', optimizer = 'adam')
model.summary()

print("# of samples:", len(samples*3*2))
print("Batch size:", batch_size)
print("# valid samples:", number_valid_steps)
print("# per epoch:", steps_per_epoch)

#model = load_model('model.h5')
history_object=model.fit_generator(train_generator,
                                   steps_per_epoch=steps_per_epoch,
                                   validation_data=validation_generator,
                                   validation_steps=number_valid_steps, 
                                   epochs=20, verbose=1)
model.save('model.h5')
                               
# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()