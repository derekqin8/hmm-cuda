# cudaHMM

## CPU Demo

The CPU Demo can be found in the `cpu` directory.

### Installation

To run the CPU demo, run

```shell
pip install -r requirements.txt
cd cpu
python3 demo.py
```

This can take several minutes to run, and should generate a series of sentences
generated using a Hidden Markov Model trained on the Bee Movie script by
default.

To train on different inputs, modify `demo.py`

### Testing

Tests were taken from CS 155 to verify correctness. Each test is in the test
directory under `cpu/tests` and labeled `test_<functionality>.py`. Simply run
each as a script. The expected output is stored in the corresponding text file.

### GPU Parallelizability

See `TODO` comments in the CPU demo source code.