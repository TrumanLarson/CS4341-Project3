from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sys
import math
import random

TRAINING_SET_SIZE = 0.6
VALIDATION_SET_SIZE = 0.15

def main(argv):
    # Model Template

    model = Sequential() # declare model
    model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('relu'))
    #
    #
    #
    # Fill in Model Here
    #
    #
    model.add(Dense(10, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))


    # Compile Model
    model.compile(optimizer='sgd',
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    # Load training data
    images = np.load('images.npy')
    labels = np.load('labels.npy')
    x_train, y_train, x_val, y_val, x_test, y_test = separateData(images, labels)

    # Train Model
    history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=10, batch_size=512)


    # Report Results

    print(history.history)
    model.predict()

def separateData(images, labels):
    resultSets = {'x_train':[], 'y_train':[], 'x_val':[], 'y_val':[], 'x_test':[], 'y_test':[]}
    organizedImages = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[]}
    for i in range(len(images)):
        organizedImages[str(labels[i])].append(np.reshape(images[i], 28*28))
    for i in organizedImages.keys():
        trainingDataCutoff = math.floor(len(organizedImages[i]) * TRAINING_SET_SIZE)
        validationDataCutoff = math.floor(len(organizedImages[i]) * VALIDATION_SET_SIZE) + trainingDataCutoff
        numImagesInClass = len(organizedImages[i])
        for j in range(numImagesInClass):
            idx = random.randint(0, (numImagesInClass - j)-1)
            if j < trainingDataCutoff:
                resultSets['x_train'].append(organizedImages[i].pop(idx))
                resultSets['y_train'].append(int(i))
            elif j < validationDataCutoff:
                resultSets['x_val'].append(organizedImages[i].pop(idx))
                resultSets['y_val'].append(int(i))
            else:
                resultSets['x_test'].append(organizedImages[i].pop(idx))
                resultSets['y_test'].append(int(i))
    return (np.array(resultSets['x_train']), np.array(resultSets['y_train']),
        np.array(resultSets['x_val']), np.array(resultSets['y_val']),
        np.array(resultSets['x_test']), np.array(resultSets['y_test']))

if __name__ == "__main__":
    main(sys.argv)