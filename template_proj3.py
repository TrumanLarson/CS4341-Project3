from keras.models import Sequential
from keras.layers import Dense, Activation
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
    #
    #
    model.add(Dense(10, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))


    # Compile Model
    model.compile(optimizer='sgd',
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    # Train Model
    history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=10, batch_size=512)


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


if __name__ == "__main__":
    main(sys.argv)
