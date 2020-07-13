"""
Script to change 'method' keys to 'eval_method'

For consistency
"""

import json

MNIST_RESULTS_DIR = "experiments/mnist/results/"
MNIST_QUICK_RESULTS_DIR = "experiments/mnist/results/quick/"


with open(MNIST_RESULTS_DIR + "all.json") as f:
    results = json.load(f)

for r in results:
    if 'method' in r:
        r['eval_method'] = r['method']
        del r['method']

with open(MNIST_RESULTS_DIR + "all.json", 'w') as f:
    json.dump(results, f,
              indent=4)

