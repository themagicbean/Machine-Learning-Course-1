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
def calculatePerformance(trueNeg,falseNeg,truePos,falsePos):
    accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
    precision = truePos / (truePos + falsePos)
    recall = truePos / (truePos + falseNeg)
    f1Score = (2 * (precision + recall)) / (precision + recall)
    return [accuracy, precision, recall, f1Score]
#works, need to get named tuples to maniputlate

my_matrix = [[67,33], [75, 25]]
calculatePerformance(my_matrix[0][0], my_matrix[0][1], my_matrix[1][0], my_matrix[1][1]) 
#works
    
