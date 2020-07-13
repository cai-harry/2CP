"""
Script to sort the keys in results jsons.
"""

import json

QUICK = False

MNIST_RESULTS_DIR = "experiments/mnist/results/"
if QUICK:
    MNIST_RESULTS_DIR += "quick/"

with open(MNIST_RESULTS_DIR + "all.json") as f:
    results = json.load(f)

with open(MNIST_RESULTS_DIR + "all.json", 'w') as f:
    json.dump(results, f,
              indent=4,
              sort_keys=True)