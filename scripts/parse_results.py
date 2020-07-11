"""
Script to combine all results into one file

Also fills in missing fields like 'method'
"""

from glob import glob
import json
import os

MNIST_RESULTS_DIR = "experiments/mnist/results/"
MNIST_QUICK_RESULTS_DIR = "experiments/mnist/results/quick/"

results = []

for filename in glob(MNIST_RESULTS_DIR + "*.json"):
    with open(filename) as f:
        result = json.load(f)
    if 'method' not in result:
        result['method'] = 'shapley'
    results.append(result)

for filename in glob(MNIST_RESULTS_DIR + "*.json"):
    os.remove(filename)

with open(MNIST_RESULTS_DIR + "all.json", 'w') as f:
    json.dump(results, f,
              indent=4)

