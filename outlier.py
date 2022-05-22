# Different simple outlier detection techniques
# By: Khaled Abdelaal
# khaled.abdelaal@ou.edu
#   Nov, 2020

import matplotlib.pyplot as plt
import math

data =[152.36, 130.38, 101.54, 96.26, 88.03, 85.66, 83.62, 76.53, 74.36, 73.87, 73.36, 73.35, 68.26, 65.25, 63.68, 63.05, 57.53]
#Plot a histogram for the data
plt.figure(1)
plt.hist(data)
plt.savefig('histogram.png')

#Scatter plot for data 
x = [y for y in range(len(data))]
plt.figure(2)
plt.plot(x,data,'o', color='black')
plt.savefig('scatter.png')

#A helper function to calculate the mean
def findMean(data):
    return sum(data)/len(data)

#A helper function to calculate standard deviation
def findStdDev(data):
    mean = findMean(data)
    diffMean = [math.pow(x - mean, 2) for x in data]
    stdDev = math.sqrt(sum(diffMean) / len(data))
    return stdDev

# A function to find outliers using Grubbs Test
# arguments:
#       dataset: the dataset
#       alpha: the value of alpha
def grubbsTest(dataset, alpha):
    # Make a copy of the dataset since this method will modify it
    # no need for deepcopy, since all list items are floats (not objects)
    data = dataset.copy()
    # Table with g critical values for two-sided tests
    # source: http://www.statistics4u.com/fundstat_eng/ee_grubbs_outliertest.html
    g_crit = {}

    g_crit[0.05] = {3:1.543, 4:1.4812, 5:1.7150, 6:1.8871, 7:2.0200, 8:2.1266, 9:2.2150, 10:2.2900, 11:2.3547
                   ,12:2.4116, 13:2.4620, 14:2.5073, 15:2.5483, 16:2.5857, 17:2.6200, 18:2.6516, 19:2.6809
                   ,20:2.7082, 25:2.8217, 30:2.9085, 40:3.0361, 50:3.1282, 60:3.1997, 70:3.2576, 80:3.3061
                   ,90:3.3477, 100:3.3841, 120:3.4451, 140:3.4951, 160:3.5373, 180:3.5736, 200:3.6055
                   ,300:3.7236, 400:3.8032, 500:3.8631, 600:3.9109}

    g_crit[0.01] = {3:1.547, 4:1.4962, 5:1.7637, 6:1.9728, 7:2.1391, 8:2.2744, 9:2.3868, 10:2.4821, 11:2.5641
                   ,12:2.26357, 13:2.6990, 14:2.7554, 15:2.8061, 16:2.8521, 17:2.8940, 18:2.9325, 19:2.9680
                   ,20:3.0008, 25:3.1353, 30:3.2361, 40:3.3807, 50:3.4825, 60:3.5599, 70:3.6217, 80:3.6729
                   ,90:3.7163, 100:3.7540, 120:3.8167, 140:3.8673, 160:3.9097, 180:3.9460, 200:3.9777
                   ,300:4.0935, 400:4.1707, 500:4.2283, 600:4.2740}

    # A flag which tells whether the dataset still has outliers or not
    # initially, we assume that it does have at least one outlier
    has_outliers = True

    while has_outliers:
        #number of elements of the dataset
        n = len(data)

        #calculate mean
        mean = findMean(data)

        #calculate standard deviation
        stdDev = findStdDev(data)

        #calculate the deviation for each point from the mean
        meanDiff = [abs(x - mean) for x in data]

        #find the data point with max deviation from the mean (i.e. max|xi - x'|
        item_with_max = data[meanDiff.index(max(meanDiff))]

        #calculated value of g
        g_calculated = max(meanDiff)/stdDev

        #get the critical value of g from the table, based on the values of alpha and n
        g_crit_val = g_crit[alpha][n]

        if (g_calculated > g_crit_val):
            # if the calculated value of g is greater than the critical value:
            # that means that there's at least one outlier
            # and the first outlier is the data point with the maximum deviation from the mean
            # so we delete it from the dataset and repeat the loop from the beginning with 
            # the modified dataset
            print("Found an outlier", item_with_max)
            data.pop(data.index(item_with_max))
            print("Outlier removed")
        else:
            has_outliers = False
            print("No more outliers!")

# A function to find outliers using parametric method 1 
# this function takes two arguments:
#       data: the dataset 
#       w   : a threshold for point deviation from the mean of the data
def parametricMethod1(data, w):
    mean = sum(data)/len(data)
    diffMean = [math.pow(x - mean,2) for x in data]
    stdDev = math.sqrt(sum(diffMean) / len(data))

    outliers = []
    for point in data:
        if abs(point-mean) / stdDev > w:
            outliers.append(point)
    print("Outliers are:" , outliers)

# A function to find outliers using K-Nearest Neighbors
# this function takes two arguments:
#       data: the dataset 
#       k   : the order of the neighbor to be used as the outlier score
#             For example, k=4 means that the distance to the 4th nearest-neighbor
#             will determine the outlier score for each data point
def kNearestNeighbor(data, k):
    # since array indexing start from 0 , we need to decrement the value of k
    k = k-1

    #a dictionary that represents distance matrix
    distances = {}

    #a dictionary to capture outlier score for each data point
    outlier_score = {}

    # for each data point, calculate the distance between this point and all other points
    # capture the results in the distances dictionary
    # for example, distance for point p1 will be distances[p1] = [d0, d1, d2, ..., dn-1]
    for i in range(len(data)):
        distances[i] = []
        for j in range(len(data)):
            if i == j:
                dist = 0
            else:
                dist = math.sqrt(math.pow(data[i] - data[j], 2))
            distances[i].append(dist)
    print("data point : outlier score")

    # Outlier score depends on the value of k
    # for example , if k=5 then outlier score is the distance to 5th nearest neighbor
    # so, for each point we sort the distance array (ascending order) 
    # then, outlier score is the distance to the kth element in the distance array for that point
    for i in range(len(data)):
        distances[i].sort()
        k_nearest_dist = distances[i][k]
        outlier_score[i] = k_nearest_dist
        print(data[i], ":", "%.2f" % outlier_score[i])



# Main program

#Parametric method 1 with w=1, 2, 3
for i in range(1,4):
    print("Parametric Method 1")
    print("==================")
    print("Using a value w =",i)
    parametricMethod1(data, i)
    print("--------------------")

#K nearest neighbors with k = 2,3
for i in range(2,4):
    print("K Nearest Neighbor with k =", i)
    kNearestNeighbor(data, i)

#Grubbs Test with alpha = 0.05
print("Grubbs Test")
grubbsTest(data, 0.05)
