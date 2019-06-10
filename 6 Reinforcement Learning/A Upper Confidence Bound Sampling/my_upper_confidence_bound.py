# Upper Confidence Bound
# lecture 170
# more with strategies here 
# https://gist.github.com/roycoding/fc430c360c87a755047185b796c10a5e

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# dataset has column for ad, row for visit, 1/0 val for clickthrough rates
# dataset is only for simulation
# in reality, start with no data.  
# so to actually apply need to be able to import and re-use data

#non-UCB algorithm = random selection file

# Implementing UCB
import math
N = 10000  # number of rounds -- need to change to some kind of input for acutal use
# *** instructor suggested spark for data streaming ****
d = 10 # number of ads (arms)
ads_selected = [] # will be set with list of ads/arms selected; starts at 0 
# step 1 in notes
numbers_of_selections = [0] * d #N(i) in notes
# brackets around 0 times d = creates vector with all zeroes 
sums_of_rewards = [0] * d # r[overbar]{sub}(i) in notes, vector with zero for each arm
total_reward = 0  
# step 2 and 3 using for loop
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0): # false if arm not used yet
            #this if condition makes sure d first rounds we just run each ad once
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            # can't use "log n" b/c indexes start at 0 
            upper_bound = average_reward + delta_i
            # don't need lower bound for this model though it is in notes
        else:
            upper_bound = 1e400 # = 10^400
            # this & next if make sure any ad (arm) d not run gets priority
        if upper_bound > max_upper_bound: # this always true initially?
            max_upper_bound = upper_bound
            ad = i # i value of specifc arm is = arm giving upper bound
    ads_selected.append(ad) # this is key piece for histogram / strategy
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    # need to change reward to some kind of input/append/stream to actually use this
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
# total reward now over 2000

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
# total reward not plotted on histogram
# remember index starts at zero so actual best ad/arm is ad/arm + 1