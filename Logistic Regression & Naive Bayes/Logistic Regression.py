# Logistic Regression classification on MNIST dataset.

import scipy.io
import numpy as np
import datetime

starttime = datetime.datetime.now()
print("\nSTARTED : ", starttime, "\n")

"""Class for logistic regression classifier."""
class LR():

    # Learn initial weights according to the training set
    def calculations(self, X_train, y_train):
        learning_rate = 0.2
        m = len(X_train)
        n = len(X_train[0])
        iterations = 100
        X_train = X_train.T
        y_train = y_train.T
        self.w = np.zeros((n,1))
        for i in range(iterations):
            dot = np.dot(self.w.T,X_train)
            sgmd_dot = self.sigmoid(dot)
            diff = y_train - sgmd_dot
            dw = np.matmul(X_train, diff.T) / m
            self.w += learning_rate * dw

    """Make prediction for the testing set."""
    def predict(self, X_test, y_test):
        m = len(X_test)
        X_test = X_test.T
        y_test = y_test.T
        dot = np.dot(self.w.T, X_test)
        sgmd_dot = self.sigmoid(dot)
        y_pred = (sgmd_dot >= 0.5)

        prediction = y_pred.T
        reality = y_test.T
        len_7 = sum([1 if reality[i] == 0 else 0 for i in range(len(reality))])
        len_8 = len(reality) - len_7
        accrcy_7 = 0
        accrcy_8 = 0
        for i in range(len(reality)):
            if (reality[i] == 1) and (prediction[i] == True):
                accrcy_8 += 1
            elif (reality[i] == 0) and (prediction[i] == False):
                accrcy_7 += 1
        print("Accuracy for digit 7: ", accrcy_7/len_7)
        print("Accuracy for digit 8: ", accrcy_8/len_8)
        print("Overall Accuracy: ", (accrcy_7 + accrcy_8)/m)

    """sigmoid function for given values"""
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    Numpyfile= scipy.io.loadmat('mnist_data.mat')
    """Retrieve training data"""
    X_train = Numpyfile['trX']
    y_train = Numpyfile['trY'].T
    ob = LR()
    ob.calculations(X_train, y_train)

    """Retrieve test data"""
    X_test = Numpyfile['tsX']
    y_test = Numpyfile['tsY'].T
    ob.predict(X_test, y_test)


endtime = datetime.datetime.now()
print("\nENDED : ", endtime)
print("\nDURATION :", endtime - starttime)