import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import hmm
import numpy as np
import time

def unsupervised_learning(n_states, N_iters,rng=np.random.RandomState(1)):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.
    Arguments:
        n_states:   Number of hidden states that the HMM should have.
        N_iters:    Number of EM steps taken.
        rng:        The random number generator used. Default to 1.
    '''
    genres, genre_map = utils.Utility.load_ron_hidden()

    # Train the HMM.
    start = time.time()
    HMM = hmm.unsupervised_HMM(genres, n_states, N_iters, rng=rng)
    print('fit time for ', N_iters, 'iterations:', time.time() - start)

    # Print the transition matrix.
    print("Transition Matrix:")
    print('#' * 70)
    for i in range(len(HMM.A)):
        print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    print('')
    print('')

    # Print the observation matrix. 
    print("Observation Matrix:  ")
    print('#' * 70)
    for i in range(len(HMM.O)):
        print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    print('')

unsupervised_learning(4, 1000,rng=np.random.RandomState(1))