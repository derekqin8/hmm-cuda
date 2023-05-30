# cudaHMM

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