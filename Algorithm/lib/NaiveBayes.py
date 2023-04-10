# nD k-Class Gausian Discriminant Analysis
import numpy as np
import math
from math import log


class NaiveBayesBernoulli():

    def __init__(self):
        pass

# Calculate gradient alpha
    def membership_function(self, test_set, alpha, priorVal):
        value_1 = []
        value_2 = []
        for x in alpha:
            if x == 0.0 or x == 1.0:
                value_1.append(0.0)
                value_2.append(0.0)
            else:
                value_1.append(log(x))
                value_2.append(log(1 - x))

        gX = [(np.sum(((test_set[:,j][i]*value_1[j]) + ((1-test_set[:,j][i])*value_2[j])) for j in range(test_set.shape[1]-1))
              + math.log(priorVal)) for i in range(test_set.shape[0])]
        return gX

# Calculate prediction
    def discriminant_function(self, max_gX, clas):

        diff = []
        predicted = []
        array = []

        for index in range(len(clas)):
            array.append(max_gX[index])
        predicted_values = np.maximum.reduce(array)

        for x in range(len(predicted_values)):
            for y in max_gX:
                key = max_gX[y]
                if(key[x] == predicted_values[x]):
                    predicted.append(y)
        return predicted

# Find precision recall and F-measure
    def findOtherParameters(self, confusion_mat):

        list_diagonal = np.zeros(confusion_mat.shape[0])
        list_row_sum = np.zeros(confusion_mat.shape[0])
        list_column_sum=np.zeros(confusion_mat.shape[1])

        precision_value = []
        recall_value = []
        f_measure_value = []

        total = np.sum(confusion_mat)
        confuse_diagonal = 0

        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                list_row_sum[i] += confusion_mat[i][j]
                list_column_sum[i] += confusion_mat[j][i]
                if(i==j):
                    list_diagonal[i] = confusion_mat[i][j]
                    confuse_diagonal +=  confusion_mat[i][j]
        # print "Accuracy", float(confuse_diagonal)/total
        accuracy = float(confuse_diagonal)/total

        for index in range(len(list_row_sum)):
            if list_row_sum[index]==0:
                precision_value.append(0.0)
            else:
                precision_value.append((float)(list_diagonal[index]) / list_row_sum[index])

            if list_column_sum[index]==0:
                recall_value.append(0)
            else:
                recall_value.append((float)(list_diagonal[index]) / list_column_sum[index])

            if precision_value[index]==0 or recall_value[index]==0:
                f_measure_value.append(0)
            else:
                f_measure_value.append((float) (2 * precision_value[index] * recall_value[index]) / (precision_value[index] + recall_value[index]))

        return accuracy, precision_value, recall_value, f_measure_value