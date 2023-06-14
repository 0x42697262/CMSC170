"""
Machine Problem 2 - Hidden Markov Model

References:
    https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e?gi=5d1e73d86e55

    T: length of the observation sequence.
    N: number of latent (hidden) states.
    M: number of observables.
    Q: {q₀, q₁, …} - hidden states.
    V: {0, 1, …, M — 1} - set of possible observations.
    A: state transition matrix.
    B: emission probability matrix.
    π: initial state probability distribution.
    O: observation sequence.
    X: (x₀, x₁, …), x_t ∈ Q - hidden state sequence.

"""
