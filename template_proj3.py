from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np
import sys

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

    x_train = np.load("images.npy")
    y_train = to_categorical(np.load("labels.npy"))

    

    #x_train = np.reshape(x_train, 28*28)
    x_train_new = []
    for i in range(len(x_train)):
        x_train_new.append(np.reshape(x_train[i], 28*28))
    x_train = x_train_new

    x_train = np.array(x_train)
    #y_train = np.reshape(y_train, -1)
    print(x_train)
    print(y_train)
    x_val = x_train
    y_val = y_train

    # TODO partition training set and validation set
    # (x_train, y_train, x_val, y_val)


    print("Training...")
    # Train Model
    history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=100, batch_size=512)


    # Report Results

    #create the matrix
    confusionMatrix = []
    for i in range(0,10):
        confusionMatrix.insert(0, [0,0,0,0,0,0,0,0,0,0])

    print(history.history)
    #modified
    col = model.predict()
    row = 0
    #row = trueval()

    confusionMatrix[row][col] += 1



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


if __name__ == "__main__":
    main(sys.argv)
