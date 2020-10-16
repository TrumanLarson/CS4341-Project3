from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np
import sys
import math
import random
import matplotlib.pyplot as plt

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
    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(Activation('sigmoid'))


    model.add(Dense(200, kernel_initializer='he_normal'))
    model.add(Activation('sigmoid'))

    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(Activation('sigmoid'))
  
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

    y_train = to_categorical(y_train)
    y_val   = to_categorical(y_val)
    y_test  = to_categorical(y_test)
    
    

 

    print("Training...")
    # Train Model
    history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=10000, batch_size=50)


    # Report Results

    #create the matrix
    confusionMatrix = []
    for i in range(0,10):
        confusionMatrix.insert(0, [0,0,0,0,0,0,0,0,0,0])

    #print(history.history)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#    x_complete = x_train
#    y_complete = y_train

 #   x_complete = np.append(x_train, x_val)
  #  y_complete = np.append(y_train, y_val)
    
    
   
    # for i in range(0,len(y_train):
        
    #     col = model.predict((x_train[i]), batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)#I think this works
    #     row = y_train[i]

    #     confusionMatrix[row][col] += 1

    # for i in range(0,len(y_val):
        
    #     col = model.predict((x_val[i]), batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)#I think this works
    #     row = y_val[i]

    #     confusionMatrix[row][col] += 1
    print(y_test[0])

    predicted = model.predict(x_test)#I think this works
    print(predicted)
    for i in range(0,len(y_test)):
        # [0, ..., 0 , 1, 0, ...]
        col = sqwish(predicted[i])
        row = sqwish(y_test[i])

        if col != row:
            print(x_test[i])

        confusionMatrix[row][col] += 1

    acc = modelAccuracy(confusionMatrix)
    
    print("Model Accuracy: ",end="")
    print(acc,end="\n")
    
    for i in range(0,10):
        
        precision = modelPrecision(confusionMatrix, i)
        
        recall = modelRecall(confusionMatrix, i)
        
        print("Model Precision for ",end="")
        print(i,end="")
        print(": ",end="")
        print(precision,end="\n")

        print("Model Recall for ",end="")
        print(i,end="")
        print(": ",end="")
        print(recall,end="\n")

    #print(confusionMatrix)

def sqwish(array):
    maxVal = -1
    maxIndex = -1
    for i in range(len(array)):        
        if array[i] > maxVal:
            maxVal = array[i]
            maxIndex = i
    return maxIndex
#functions for analyzing success of the model
def modelAccuracy(matrix):
    """
    Return the accuracy of the model based on the confusion matrix.  
    
    Parameters
    ----------
    matrix : list of lists
        The confusion matrix for the model
    """   
    correct = 0
    total = 0
    for i in range(0,10):
        correct += matrix[i][i]
        for j in range(0,10):
            total += matrix[i][j]
    return (correct/total)


def modelPrecision(matrix, predictedVal):
    """
    Return the precision of the model for a particular value based on the confusion matrix.  
    
    Parameters
    ----------
    matrix : list of lists
        The confusion matrix for the model
    predictedVal: integer 0 to 9
        The value predicted by the model
    """   
    correct = matrix[predictedVal][predictedVal]
    total = 0
    for n in range(0,10):
        total += matrix[n][predictedVal]
        
    if total == 0:
        return 0
    else:
        return (correct/total)


def modelRecall(matrix, trueVal):
    """
    Return the recall of the model for a particular value based on the confusion matrix.  
    
    Parameters
    ----------
    matrix : list of lists
        The confusion matrix for the model
    trueVal: integer 0 to 9
        The true value of the test by the model
    """   
    correct = matrix[trueVal][trueVal]
    total = 0
    for n in range(0,10):
        total += matrix[trueVal][n]

    if total == 0:
        return 0
    else:    
        return (correct/total)


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
