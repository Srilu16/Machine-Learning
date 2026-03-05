import numpy as np
import pandas as pd

def binning(column, bins=4, method="width"):

    column = np.array(column)

    if method == "width":
        min_val = column.min()
        max_val = column.max()

        bin_width = (max_val - min_val) / bins

        binned = []
        for value in column:
            index = int((value - min_val) / bin_width)
            if index == bins:
                index = bins - 1
            binned.append(index)

        return np.array(binned)

    elif method == "frequency":
        sorted_index = np.argsort(column)
        binned = np.zeros(len(column))

        size = len(column) // bins

        for i in range(bins):
            start = i * size
            end = (i + 1) * size if i != bins-1 else len(column)

            for j in sorted_index[start:end]:
                binned[j] = i

        return binned

df = pd.read_csv("dataset.csv")

feature = df.iloc[:,0]   

bins_default = binning(feature)

bins_width = binning(feature, bins=5, method="width")

bins_freq = binning(feature, bins=5, method="frequency")

print("Equal Width Binning:", bins_width[:10])
print("Equal Frequency Binning:", bins_freq[:10])
