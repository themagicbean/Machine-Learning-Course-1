# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
# --note hard coded not imported -- look for and scrutinize libraries?
import random #instead of math (used in UCB)
N = 10000
d = 10
ads_selected = []
#now changing parameters of UCB to TS parameters
numbers_of_rewards_1 = [0] * d # number of times reward 1
numbers_of_rewards_0 = [0] * d # or 0
# [0] * d initializes a vector with d elements and all values 0
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        # random.betavariate gives variates of x.y with parameters given
        # beta distribution is ... ?
        
        if random_beta > max_random: # this basically same as UCB w changed var names
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:  # this if/else updates rewards of ads
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    #tighter solution
        #nor 1 += reward
        #nor 0 += 1- reward
    total_reward = total_reward + reward
  
# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()