import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle


def adjust_gamma(image):
    # randomly adjust gamma value
    gamma = np.random.rand() + 0.5
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted  gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


#correction = 0.3    # correction for steer
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.15    # correction for steer
    cor_index = [0, 1., -1.]

    while 1:    # Loop forever so the generator never terminate
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3]) + cor_index[i] * correction
                    images.append(image)
                    angles.append(angle)
                    # adjust gamma
                    images.append(adjust_gamma(image))
                    angles.append(angle)

            aug_images = []
            aug_angles = []
            for image, angle in zip(images, angles):
                aug_images.append(image)
                aug_angles.append(angle)
                aug_images.append(cv2.flip(image, 1))
                aug_angles.append(angle * -1.)

            # could add other prepressing (trim image to only see section with road)
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
valid_generator = generator(validation_samples, batch_size=32)


# build a simple neural netowrk first
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50, 25), (0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# train model and plot the loss
adam = optimizers.Adam() # lr=0.0008
model.compile(loss='mse', optimizer=adam)
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                        validation_data=valid_generator, nb_val_samples=len(validation_samples),
                        nb_epoch=25, verbose=1)

model.save('model.h5')

# print the key contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training_set', 'validation_set'], loc='upper right')
plt.show()

