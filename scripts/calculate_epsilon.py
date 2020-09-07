"""
A convenience script that uses pyvacy.analysis to print epsilon value for the given parameters.

Used to pick parameters for experiments.
"""


import pyvacy
from pyvacy.analysis import moments_accountant as epsilon

TRAINING_ITERATIONS = 16
TRAINING_HYPERPARAMS = {
    'final_round_num': TRAINING_ITERATIONS,
    'batch_size': 8,
    'epochs': 16,
    'learning_rate': 1e-2
}
DP_PARAMS = {
    'l2_norm_clip': 1.0,
    'noise_multiplier': 1.5,
    'delta': 1e-5
}

DATA_LENGTH = 15000

eps = epsilon(
    N=DATA_LENGTH,
    batch_size=TRAINING_HYPERPARAMS['batch_size'],
    noise_multiplier=DP_PARAMS['noise_multiplier'],
    epochs=TRAINING_HYPERPARAMS['epochs'],
    delta=DP_PARAMS['delta']
)

print(f"eps={eps}, delta={DP_PARAMS['delta']}")