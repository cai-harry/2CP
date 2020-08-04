import argparse

from experiments.mnist.mnist import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments on the COVID dataset using 2CP.")
    parser.add_argument(
        '--full',
        help='Do a full run. Otherwise, does a quick run for testing purposes during development.',
        action='store_true'
    )
    args = parser.parse_args()

    QUICK_RUN = not args.full
    if QUICK_RUN:
        TRAINING_ITERATIONS = 1
    else:
        TRAINING_ITERATIONS = 16
    TRAINING_HYPERPARAMS = {
        'final_round_num': TRAINING_ITERATIONS,
        'batch_size': 8,
        'epochs': 10,
        'learning_rate': 1e-2
    }
    ROUND_DURATION = 1800  # should always end early

    if QUICK_RUN:
        experiments = [
            {
                'dataset': 'covid',
                'split_type': 'dp',
                'num_trainers': 3,
                'dp_params': {
                    'l2_norm_clip': 1.0,
                    'noise_multiplier': 1.1,
                    'delta': 1e-5
                }
            },
        ]
        method = 'step'
        seed = 88
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                run_experiment(protocol=protocol,
                               eval_method=method, seed=seed, **exp)
    else:
        experiments = [
            {'dataset': 'covid', 'split_type': 'noniid', 'num_trainers': 3, 'disjointness': 0.0},
            {'dataset': 'covid', 'split_type': 'noniid', 'num_trainers': 3, 'disjointness': 0.2},
            {'dataset': 'covid', 'split_type': 'noniid', 'num_trainers': 3, 'disjointness': 0.4},
            {'dataset': 'covid', 'split_type': 'noniid', 'num_trainers': 3, 'disjointness': 0.6},
            {'dataset': 'covid', 'split_type': 'noniid', 'num_trainers': 3, 'disjointness': 0.8},
            {'dataset': 'covid', 'split_type': 'noniid', 'num_trainers': 3, 'disjointness': 1.0},
        ]
        method = 'step'
        seed = 89
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                print(f"Starting experiment with args: {exp}")
                run_experiment(protocol=protocol,
                               eval_method=method, seed=seed, **exp)