import random
import numpy as np


class HiddenMarkovModel:
    """
    Class implementation of Hidden Markov Models.
    """

    def __init__(self, A, O):
        """
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.
        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        """

        self.A = np.array(A)
        self.O = np.array(O)
        self.L = self.A.shape[0]
        assert self.A.shape[0] == self.A.shape[1]

        self.D = self.O.shape[1]
        assert self.O.shape[0] == self.L

        self.A_start = np.ones(self.L) / self.L

    def viterbi(self, x):
        """
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        """

        M = len(x)  # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0.0 for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [["" for _ in range(self.L)] for _ in range(M + 1)]

        for a in range(self.L):
            probs[1][a] = self.O[a][x[0]] * self.A_start[a]
            seqs[1][a] += str(a)

        for j in range(2, M + 1):
            # compute each next y_a
            for a in range(self.L):
                max_prob = 0
                prev = ""
                # iterate through previous sequences, this is y^{j-1}
                for b in range(self.L):
                    prob = probs[j - 1][b] * self.A[b][a] * self.O[a][x[j - 1]]

                    if prob > max_prob:
                        max_prob = prob
                        prev = seqs[j - 1][b]

                probs[j][a] = max_prob
                seqs[j][a] = prev + str(a)

        max_seq = seqs[M][probs[M].index(max(probs[M]))]
        return max_seq

    def forward(self, x, normalize=False):
        """
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.
                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        """

        M = len(x)  # Length of sequence.
        alphas = [[0.0 for _ in range(self.L)] for _ in range(M + 1)]

        for a in range(self.L):
            alphas[1][a] = self.O[a][x[0]] * self.A_start[a]

        for t in range(2, M + 1):
            for a in range(self.L):
                tot = 0
                for b in range(self.L):
                    tot += self.A[b][a] * self.O[a][x[t - 1]] * alphas[t - 1][b]

                alphas[t][a] = tot

            if normalize:
                C = sum(alphas[t])
                alphas[t] = [alpha / C for alpha in alphas[t]]

        return alphas

    def backward(self, x, normalize=False):
        """
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        """

        M = len(x)  # Length of sequence.
        betas = [[0.0 for _ in range(self.L)] for _ in range(M + 1)]

        for a in range(self.L):
            betas[M][a] = 1.0

        for t in range(M - 1, 0, -1):
            for a in range(self.L):
                tot = 0
                for b in range(self.L):
                    tot += self.A[a][b] * self.O[b][x[t]] * betas[t + 1][b]

                betas[t][a] = tot

            if normalize:
                C = sum(betas[t])
                betas[t] = [beta / C for beta in betas[t]]

        return betas

    def supervised_learning(self, X, Y):
        """
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        """

        N = len(X)

        # Calculate each element of A using the M-step formulas.

        for a in range(self.L):
            for b in range(self.L):
                numerator = 0
                denominator = 0
                for i in range(N):
                    for j in range(1, len(Y[i])):
                        if Y[i][j] == b and Y[i][j - 1] == a:
                            numerator += 1
                        if Y[i][j - 1] == a:
                            denominator += 1
                self.A[a][b] = numerator / denominator

        # Calculate each element of O using the M-step formulas.

        for a in range(self.L):
            for w in range(self.D):
                numerator = 0
                denominator = 0
                for i in range(N):
                    for j in range(len(Y[i])):
                        if X[i][j] == w and Y[i][j] == a:
                            numerator += 1
                        if Y[i][j] == a:
                            denominator += 1
                self.O[a][w] = numerator / denominator

    def unsupervised_learning(self, X, N_iters):
        """
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
            N_iters:    The number of iterations to train on.
        """

        N = len(X)

        # j is 0 indexed for these 2 arrays

        # marginal_state[i][j][a] = P(y^j = a, x_i)
        marginal_state = [
            [[0.0 for _ in range(self.L)] for _ in range(len(x))] for x in X
        ]
        # marginal_trans[i][j][a][b] = P(y^{j} = a, y^{j+1} = b, x_i)
        # starts at 1 and ends at M-1
        marginal_trans = [
            [
                [[0.0 for _ in range(self.L)] for _ in range(self.L)]
                for _ in range(len(x))
            ]
            for x in X
        ]

        for epoch in range(N_iters):
            for i in range(len(X)):
                # alphas[j][a] = alpha_a(j)
                alphas = self.forward(X[i], True)
                # betas[j][a] = beta_a(j)
                betas = self.backward(X[i], True)

                for j in range(1, len(X[i]) + 1):
                    denominator = sum(
                        [
                            alphas[j][a_prime] * betas[j][a_prime]
                            for a_prime in range(self.L)
                        ]
                    )
                    for a in range(self.L):
                        marginal_state[i][j - 1][a] = (
                            alphas[j][a] * betas[j][a]
                        ) / denominator

                for j in range(1, len(X[i])):
                    denominator = sum(
                        [
                            sum(
                                [
                                    alphas[j][a_prime]
                                    * betas[j + 1][b_prime]
                                    * self.O[b_prime][X[i][j]]
                                    * self.A[a_prime][b_prime]
                                    for b_prime in range(self.L)
                                ]
                            )
                            for a_prime in range(self.L)
                        ]
                    )
                    for a in range(self.L):
                        for b in range(self.L):
                            marginal_trans[i][j - 1][a][b] = (
                                alphas[j][a]
                                * betas[j + 1][b]
                                * self.O[b][X[i][j]]
                                * self.A[a][b]
                            ) / denominator

            # Calculate each element of A using the M-step formulas.

            for a in range(self.L):
                for b in range(self.L):
                    numerator = 0
                    denominator = 0
                    for i in range(N):
                        for j in range(len(X[i]) - 1):
                            numerator += marginal_trans[i][j][a][b]
                            denominator += marginal_state[i][j][a]

                    self.A[a][b] = numerator / denominator

            # Calculate each element of O using the M-step formulas.

            for a in range(self.L):
                for w in range(self.D):
                    numerator = 0
                    denominator = 0
                    for i in range(N):
                        for j in range(len(X[i])):
                            if X[i][j] == w:
                                numerator += marginal_state[i][j][a]
                            denominator += marginal_state[i][j][a]

                    self.O[a][w] = numerator / denominator

    def generate_emission(self, M):
        """
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        """

        emission = []
        states = []

        # choose a random start state
        states.append(random.randrange(self.L))
        emission.append(
            random.choices(range(self.D), weights=self.O[states[0]])[0]
        )

        for j in range(1, M):
            states.append(
                random.choices(range(self.L), weights=self.A[states[j - 1]])[0]
            )
            emission.append(
                random.choices(range(self.D), weights=self.O[states[j]])[0]
            )

        return emission, states

    def probability_alphas(self, x):
        """
        Finds the maximum probability of a given input sequence using
        the forward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        """

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        """
        Finds the maximum probability of a given input sequence using
        the backward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        """

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum(
            [
                betas[1][j] * self.A_start[j] * self.O[j][x[0]]
                for j in range(self.L)
            ]
        )

        return prob


def supervised_HMM(X, Y):
    """
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    """
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM


def unsupervised_HMM(X, n_states, N_iters, rng=np.random.RandomState(1)):
    """
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
        rng:        The random number generator for reproducible result.
                    Default to RandomState(1).
    """

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[rng.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[rng.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
