import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import cuhmm
import time


def sequence_prediction(n):
    """
    Runs sequence prediction on the five sequences at the end of the file
    'sequence_data<n>.txt' for a given n and prints the results.
    Arguments:
        n:          Sequence index.
    """
    A, O, seqs = utils.Utility.load_sequence(n)

    # Print file information.
    print("File #{}:".format(n))
    print(
        "{:30}{:30}".format(
            "Emission Sequence", "Max Probability State Sequence"
        )
    )
    print("#" * 70)

    # For each input sequence:
    for seq in seqs:
        # Initialize an HMM.
        HMM = cuhmm.CuHiddenMarkovModel(A, O)

        # Make predictions.
        x = "".join([str(xi) for xi in seq])
        y = HMM.viterbi(seq)

        # Print the results.
        print("{:30}{:30}".format(x, y))

    print("")


for n in range(6):
    start = time.time()
    sequence_prediction(n)
    print("Time to run: " + str(time.time() - start))
    print("")
