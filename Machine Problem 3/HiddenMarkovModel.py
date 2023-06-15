import numpy as np

Q = ['Rainy', 'Sunny']
V = ['Walk', 'Shop', 'Clean']
N = len(Q)
M = len(V)

A = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

B = np.array([
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
])

π = np.array([0.6, 0.4])

# Calculate the probability of the event "Walk" using the forward algorithm
def forward_algorithm(observation_sequence):
    T = len(observation_sequence)
    alpha = np.zeros((T, N))

    # Initialization
    alpha[0] = π * B[:, V.index(observation_sequence[0])]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.dot(alpha[t - 1], A[:, j]) * B[j, V.index(observation_sequence[t])]

    return alpha

observation_sequence = ['Walk']
alpha = forward_algorithm(observation_sequence)

# Calculate the probability of the event "Walk"
probability_walk = np.sum(alpha[-1])

print("Probability of Walk:", probability_walk)
