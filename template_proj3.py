from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
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
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
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

    y_train = to_categorical(y_train)
    y_val   = to_categorical(y_val)
    y_test  = to_categorical(y_test)
    
    

    #x_train = np.reshape(x_train, 28*28)
    x_train_new = []
    for i in range(len(x_train)):
        x_train_new.append(np.reshape(x_train[i], 28*28))
    x_train = x_train_new
    
    x_val_new = []
    for i in range(len(x_val)):
        x_val_new.append(np.reshape(x_val[i], 28*28))
    x_val = x_val_new
    
    x_test_new = []
    for i in range(len(x_test)):
        x_test_new.append(np.reshape(x_test[i], 28*28))
    x_test = x_test_new

    x_train = np.array(x_train)
    x_val   = np.array(x_val)
    x_test  = np.array(x_test)

    print("Training...")
    # Train Model
    history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=100, batch_size=512)


    # Report Results

    #create the matrix
    confusionMatrix = []
    for i in range(0,10):
        confusionMatrix.insert(0, [0,0,0,0,0,0,0,0,0,0])

    print(history.history)


    x_complete = x_train.concatenate(x_val)
    y_complete = y_train.concatenate(y_val)
    
    
    for i in range(0,len(y_complete)):
        
        col = model.predict((x_complete[i]), 1, 0, 1, true)#I think this works
        row = y_complete[i]

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
    for i in range(0,10):
        total += matrix[i][predictedVal]
        
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
    for i in range(0,10):
        total += matrix[trueVal][i]
        
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
