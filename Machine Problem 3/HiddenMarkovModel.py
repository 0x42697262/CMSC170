Q = ['rainy', 'sunny']
V = ['walk', 'shop', 'clean']
N = len(Q)
M = len(V)

A = [
    [0.7, 0.3],
    [0.4, 0.6]
]

B = [
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
]

π = [0.6, 0.4]

# Calculate the probability of the event "Walk" using the forward algorithm
def forward_algorithm(observation_sequence):
    T = len(observation_sequence)
    alpha = [[0] * N for _ in range(T)]

    # Initialization
    for j in range(N):
        alpha[0][j] = π[j] * B[j][V.index(observation_sequence[0])]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            sum_term = 0
            for i in range(N):
                sum_term += alpha[t - 1][i] * A[i][j]
            alpha[t][j] = sum_term * B[j][V.index(observation_sequence[t])]

    return alpha



O = input("Observation Sequence (separated by space): ")


observation_sequence = O.lower().split()

alpha = forward_algorithm(observation_sequence)

# Calculate the probability of the event
probability = sum(alpha[-1])

print(f"Probability of '{O}':", probability)
