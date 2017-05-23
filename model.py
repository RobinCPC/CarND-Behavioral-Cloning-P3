import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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
def collect_filelist(direct='./data/driving_log.csv', steer_alter=0.15):
    '''
    :type direct: a string of location of log file
    :type steer_alter: float number to alter steer angle
    :rtype: ( a list of file, a list of steer angle
    '''
    lines = []
    with open(direct) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    files = []
    angles = []
    cor_index = [0, 1., -1.]
    for line in lines:
        for i in range(3):
            name = './data/IMG/' + line[i].split('/')[-1]
            angle = float(line[3]) + cor_index[i] * steer_alter
            files.append(name)
            angles.append(angle)
    return files, angles


def generator(samples, labels, batch_size=32):
    assert len(samples) == len(labels), "The number of samples and labels are not equal."
    num_samples = len(samples)

    while 1:    # Loop forever so the generator never terminate
        samples, labels = shuffle(samples, labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]
            batch_labels = labels[offset : offset+batch_size]

            images = []
            angles = []
            for filename, label in zip(batch_samples, batch_labels):
                image = cv2.imread(filename)
                images.append(image)
                angles.append(label)
                # adjust gamma
                images.append(adjust_gamma(image))
                angles.append(label)

            aug_images = []
            aug_angles = []
            for image, angle in zip(images, angles):
                aug_images.append(image)
                aug_angles.append(angle)
                #aug_images.append(cv2.flip(image, 1))
                #aug_angles.append(angle * -1.)

            # could add other prepressing (trim image to only see section with road)
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Collect all file path, and its target angles
filelist, steer_angles = collect_filelist()
X_train, X_val, y_train, y_val = train_test_split(filelist, steer_angles, test_size=0.1)

# compile and train the model using the generator function
BATCH_SIZE = 128
train_generator = generator(X_train, y_train, batch_size=BATCH_SIZE)
valid_generator = generator(X_val, y_val, batch_size=BATCH_SIZE)


# build a simple neural netowrk first
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_model():
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
    adam = optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)
    model.compile(loss='mse', optimizer=adam)
    return model

model = create_model()
#model = load_model('../model_e15.h5')
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)
history_object = model.fit_generator(train_generator,
                                    samples_per_epoch=len(X_train),
                                    validation_data=valid_generator,
                                    nb_val_samples=len(X_val),
                                    nb_epoch=5, verbose=1)

model_name = 'model'
model.save('./{}.h5'.format(model_name))
#model.save_weights('./{}_weights.h5'.format(model_name))

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

