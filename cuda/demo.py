import cuhmm
import utils
import time
import datetime

with open("data/bee_movie.txt") as f:
    text = f.read()

obs, obs_map = utils.parse_observations(text)

start_time = time.time()

cuhmm = cuhmm.unsupervised_HMM(obs, 10, 100)

print(
    "Time to train: {}".format(
        str(datetime.timedelta(seconds=time.time() - start_time))
    )
)

print("Sample Sentences:")
print("=================")

for i in range(25):
    print(utils.sample_sentence(cuhmm, obs_map, n_words=25))
