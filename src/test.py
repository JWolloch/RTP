import numpy as np

arr = np.array([10.34, 557.23, 12.39, 4.00, 78.9])
mask = arr >= 11

masked_arr_indices = np.where(mask)[0]

masked_arr = arr[masked_arr_indices]

n = min(2, masked_arr.shape[0])

top_n_indices = np.argsort(masked_arr)[-n:]

most_violated_indices = masked_arr_indices[top_n_indices]

print(most_violated_indices)
print(arr[most_violated_indices])