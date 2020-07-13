"""
Script to add the training hyperparams used in each experiment
"""

import json

QUICK_RUN = False

# Copied from experiments/mnist/mnist.py
if QUICK_RUN:
    TRAINING_ITERATIONS = 2
else:
    TRAINING_ITERATIONS = 3
TRAINING_HYPERPARAMS = {
    'final_round_num': TRAINING_ITERATIONS,
    'batch_size': 32,
    'epochs': 1,
    'learning_rate': 1e-2
}

MNIST_RESULTS_DIR = "experiments/mnist/results/"
if QUICK_RUN:
    MNIST_RESULTS_DIR += "quick/"


with open(MNIST_RESULTS_DIR + "all.json") as f:
    results = json.load(f)

for r in results:
    r['quick_run'] = QUICK_RUN
    r.update(TRAINING_HYPERPARAMS)


with open(MNIST_RESULTS_DIR + "all.json", 'w') as f:
    json.dump(results, f,
              indent=4)

