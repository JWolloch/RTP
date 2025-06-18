import numpy as np

phi = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

gamma_matrix = np.array([[0.01, 0.02, 0.03, 0.04, 0.05],
                        [0.02, 0.03, 0.04, 0.05, 0.06],
                        [0.03, 0.04, 0.05, 0.06, 0.07],
                        [0.04, 0.05, 0.06, 0.07, 0.08],
                        [0.05, 0.06, 0.07, 0.08, 0.09]])

delta = 0.25

phi_initial_lower_bound = np.maximum(phi - delta, 0)

print(phi_initial_lower_bound)

phi_underbar_1 = np.max(phi_initial_lower_bound[:, None] - gamma_matrix, axis=0)
phi_bar_1 = np.min(phi_initial_lower_bound[:, None] + gamma_matrix, axis=0)

print(phi_initial_lower_bound[:, None])

print(phi_underbar_1)
print(phi_bar_1)

a = np.array([1, 2, 3, 4, 5])
A = np.diag(a)
print(A)

print(A.shape)