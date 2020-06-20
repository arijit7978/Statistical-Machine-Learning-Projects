"""
K-Means on MNIST dataset.
"""

import scipy.io as sio
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Kmeans():
    """Class for K-Means"""

    def __init__(self):
        self.mat = sio.loadmat('AllSamples', squeeze_me=True)
        self.dataSet = self.mat['AllSamples']
        self.clusterSize = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.df = pd.DataFrame(self.dataSet)
        self.df = self.df.rename(columns={0: "X", 1: "Y"})
        self.data = np.array(list(zip(self.df['X'].values, self.df['Y'].values)))

    def Euclidean(self, centers, dataPoint):
        """calculate distance between 2 points"""
        return sum([(centers[i] - dataPoint[i]) ** 2 for i in range(len(centers))]) ** 0.5

    def Member(self,centers, dataPoint):
        """choose member and calculate loss."""
        dicti = {}
        for i in range(len(centers)):
            dicti[self.Euclidean(centers[i], dataPoint)] = i
        dist = sorted(dicti.keys())[0]
        memberShip = dicti[dist]
        return memberShip, dist ** 2

    def K_Centers(self, k, strategy):
        """initialize K centers."""
        if strategy == 1:
            Indices = random.sample(range(len(self.dataSet)), k)
            centroids = [self.dataSet[index] for index in Indices]
            return centroids
        elif strategy == 2:
            centroids = []
            p = self.data[random.randint(0, len(self.data) - 1)]
            x = p[0]
            y = p[1]
            centroids.append((x, y))

            for i in range(2, k + 1):
                max_ = float('-inf')
                for i in range(len(self.data)):
                    x = self.data[i][0]
                    y = self.data[i][1]
                    dist = 0
                    if (x, y) not in centroids:
                        for centroid in centroids:
                            dist += self.Euclidean(centroid, (x, y))
                        if (max_ < dist):
                            max_ = dist
                            point = (x, y)
                centroids.append(point)
            return centroids

    def updateCenters(self, centers,  dataMship):
        """Update cntroids """
        newCenters = [] * len(centers)
        for i in range(len(centers)):
            indices = [index for index, x in enumerate(dataMship) if x == i]
            members = [dataset for index, dataset in enumerate(self.dataSet) if index in indices]
            if len(members) == 0:
                newCenter = 0
            else:
                newCenter = (sum(members)) / float(len(members))
            newCenters.append(newCenter)
        return newCenters

    def compareCenters(self, oldCenters, newCenters):
        """compare all the new and old centroids"""
        Count = 0
        for i in range(len(oldCenters)):
            Diff = self.Euclidean(oldCenters[i], newCenters[i])
            if Diff == 0:
                Count += 1
            if Count == len(oldCenters):
                return True
        return False

    def Loss_Function(self, strategy):
        """calculate objective function."""
        lossFunction = []
        for k in self.clusterSize:
            centers = self.K_Centers(k, strategy)
            oldCenters = centers

            while True:
                dataMemships, dataDistances = [], []
                for datapoint in self.dataSet:
                    memShip, distance = self.Member(oldCenters, datapoint)
                    dataMemships.append(memShip)
                    dataDistances.append(distance)
                newCenters = self.updateCenters(oldCenters, dataMemships)
                if self.compareCenters(oldCenters, newCenters):
                    lossFunction.append(sum(dataDistances))
                    break
                oldCenters = newCenters
        print(lossFunction)
        return lossFunction


    def plot_Size_Vs_Loss(self,lossFunction):
        """plot clustersize vs objective function."""
        plt.plot(self.clusterSize, lossFunction)
        plt.xlabel("Cluster Size")
        plt.ylabel("Loss Function")
        plt.title("K Means plot (Cluster Size VS Loss Function")
        plt.show()


if __name__ == "__main__":

    strategy = int(input("Enter 1 for startegy1 (random selection) or 2 for strategy2 (max avearge distance): "))
    kmn = Kmeans()

    if (strategy == 1) or (strategy == 2):
        loss = kmn.Loss_Function(strategy)
        kmn.plot_Size_Vs_Loss(loss)
    else:
        print("Enter either 1 or 2")