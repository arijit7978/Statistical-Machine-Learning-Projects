"""
Naive Bayes classification on MNIST dataset.
"""

import scipy.io
import numpy as np
import datetime

starttime = datetime.datetime.now()
print("\nSTARTED : ", starttime, "\n")

class NaiveBayes:
    """Class for Naive Bayes classifier."""
    def mean_stdv(self, training):
        mean, stdv = [], []
        training = [[training[j][i] for j in range(len(training))] for i in range(len(training[0]))]
        for val in training:
            mean.append(np.mean(val))
            stdv.append(np.std(val))
        return mean, stdv

    def calculations(self, Numpyfile):
        """Calculations of mean, standard deviation and prior probabilities"""
        self.trX = Numpyfile['trX']
        self.trY = Numpyfile['trY'].T
        self.tsX = Numpyfile['tsX']
        self.tsY = Numpyfile['tsY'].T
        trainingX_7, trainingX_8, trainingY_7, trainingY_8 = [], [], [], []

        for i in range(len(self.trY)):
            if self.trY[i]:
                trainingX_8.append(self.trX[i])
                trainingY_8.append(self.trY[i])
            else:
                trainingX_7.append(self.trX[i])
                trainingY_7.append(self.trY[i])

        self.trainingX_7_mean, self.trainingX_7_stdv = self.mean_stdv(trainingX_7)
        self.trainingX_8_mean, self.trainingX_8_stdv = self.mean_stdv(trainingX_8)

        self.prior_Y_7 = np.log(len(trainingY_7) / (len(trainingY_7) + len(trainingY_8)))
        self.prior_Y_8 = np.log(len(trainingY_8) / (len(trainingY_7) + len(trainingY_8)))

    def pdf(self, x, mean, var):
        """pdf calculation"""
        if not var:
            return 0.0001
        numerator = np.exp(- (x - mean) ** 2 / (2 * (var**2)))
        denominator = np.sqrt(2 * np.pi) * var
        ans = numerator / denominator
        if not (ans > 0.0):
            return 0.0001
        return ans

    def predict(self):
        """prediction and accuracy"""
        count = 0
        count_7 = 0
        count_8 = 0
        len_7 = sum([1 if self.tsY[i] == 0 else 0 for i in range(len(self.tsY))])
        len_8 = len(self.tsY) - len_7
        for row in range(len(self.tsX)):
            prob_7 = self.prior_Y_7
            prob_8 = self.prior_Y_8
            for col in range(len(self.tsX[0])):
                prob_7 += np.log(self.pdf(self.tsX[row][col], self.trainingX_7_mean[col], self.trainingX_7_stdv[col]))
                prob_8 += np.log(self.pdf(self.tsX[row][col], self.trainingX_8_mean[col], self.trainingX_8_stdv[col]))
            if prob_8 >= prob_7:
                if self.tsY[row]:
                    count_8 += 1
                    count += 1
            else:
                if not self.tsY[row]:
                    count_7 += 1
                    count += 1

        print("Accuracy for digit 7: ", count_7 / len_7)
        print("Accuracy for digit 8: ", count_8 / len_8)
        print("Overall Accuracy: ", count / len(self.tsX))


if __name__ == '__main__':
    Numpyfile= scipy.io.loadmat('mnist_data.mat')
    ob = NaiveBayes()
    ob.calculations(Numpyfile)
    ob.predict()

endtime = datetime.datetime.now()
print("\nENDED : ", endtime)
print("\nDURATION :", endtime - starttime)






