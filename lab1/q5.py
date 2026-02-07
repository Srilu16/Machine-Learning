import numpy as np
nums = np.random.randint(1, 11, 25)
print("List:", nums)
mean = np.mean(nums)
median = np.median(nums)
v,c = np.unique(nums, return_counts=True)
max_count = np.max(c)
mode = v[c== max_count]
print("Mean =", mean)
print("Median =", median)
print("Mode =", mode)
