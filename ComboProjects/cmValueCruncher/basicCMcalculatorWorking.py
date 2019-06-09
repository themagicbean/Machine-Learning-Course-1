# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:02:27 2018

@author: USER
"""

"""
function to output a matrix (as list) for processing LOL highlighting
"list of lists" will work per numpy

my_matrix = function(outputlistoflists)
test matrices

then compare them
"""
from my_natural_language_processing import NLPNB 

def calculatePerformance(trueNeg,falseNeg,truePos,falsePos):
    accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
    precision = truePos / (truePos + falsePos)
    recall = truePos / (truePos + falseNeg)
    f1Score = (2 * (precision + recall)) / (precision + recall)
    return [["accuracy", accuracy],["precision", precision],["recall", recall],["f1Score", f1Score]]
#works, 

CM1 = NLPNB
CM2 = [[40, 60], [80, 20]]
#calculatePerformance(my_matrix[0][0], my_matrix[0][1], my_matrix[1][0], my_matrix[1][1]) 
#works
performanceOfCM1 = calculatePerformance(CM1[0][0], CM1[0][1], CM1[1][0], CM1[1][1]) 

performanceOfCM2 = calculatePerformance(CM2[0][0], CM2[0][1], CM2[1][0], CM2[1][1]) 

print("The performance of CM1 is:", performanceOfCM1)
print("The performance of CM2 is:", performanceOfCM2)

# works@