"""
Script to add missing params to past experiments

Updated to the latest thing each time
"""

import json

QUICK_RUN = False


MNIST_RESULTS_DIR = "experiments/mnist/results/"
if QUICK_RUN:
    MNIST_RESULTS_DIR += "quick/"


with open(MNIST_RESULTS_DIR + "all.json") as f:
    results = json.load(f)

for r in results:
    r['dataset'] = 'mnist'


with open(MNIST_RESULTS_DIR + "all.json", 'w') as f:
    json.dump(results, f,
              indent=4)

