# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:31:09 2018

@author: USER
"""

def addition(a, b):
    sumofthem = a + b 
    return sumofthem
# works
    
def subtraction (a,b):
    subofthem = a - b
    return subofthem
#works
    
def multiplication (a, b):
    productofthem = a * b
    return productofthem
#works
    
def division(a,b):
    quotientofthem = (a/b)
    return quotientofthem
#works
    
# random test numbers
x = 27
y = -(1/3)

# define vars as returns on functions gets returns as vars

# is there a better way?  IDK

additval = addition(x, y)
subtraval = subtraction(x, y)
multival = multiplication(x, y)
divival = division(x,y)

values = [
        (additval, "added"),
        (subtraval, "subtracted"),
        (multival, "multplied"),
        (divival, "divided")
        ]

# works

print(values) # works

values.sort() # gotta keep id of values tho (hence complications above)
# this sorts by the ID of the tuple alphabetically if names are first
# if values first, sorts low to high

print(values)
# works!


# next steps
# break into separate files
# import data
# conquer the world
    