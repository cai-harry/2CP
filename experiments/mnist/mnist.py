import argparse

from experiments.runner import ExperimentRunner

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run experiments on the MNIST dataset using 2CP.")
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
        TRAINING_ITERATIONS = 5
    TRAINING_HYPERPARAMS = {
        'final_round_num': TRAINING_ITERATIONS,
        'batch_size': 32,
        'epochs': 1,
        'learning_rate': 1e-2
    }
    DP_PARAMS = {
        'l2_norm_clip': 1.0,
        'noise_multiplier': 1.1,
        'delta': 1e-5
    }
    ROUND_DURATION = 1800  # should always end early
 
    runner = ExperimentRunner(
        QUICK_RUN,
        TRAINING_ITERATIONS,
        TRAINING_HYPERPARAMS,
        ROUND_DURATION
    )

    if QUICK_RUN:
        experiments = []
        method = 'step'
        seed = 88
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                runner.run_experiment(protocol=protocol,
                               eval_method=method, seed=seed, **exp)
    else:
        experiments = [
            # Test A
            {'dataset': 'mnist', 'split_type': 'equal', 'num_trainers': 4},

            # Test B
            {'dataset': 'mnist', 'split_type': 'size', 'num_trainers': 4, 'ratios': [1,2,3,4]},

            # Test C
            {'dataset': 'mnist', 'split_type': 'flip', 'num_trainers': 4, 'flip_probs': [0,0.1,0.2,0.3]},

            # Test D
            {'dataset': 'mnist', 'split_type': 'noniid', 'num_trainers': 4, 'disjointness': 0.6},

            # Test E
            {
                'dataset': 'mnist',
                'split_type': 'dp',
                'num_trainers': 4,
                'dp_params': DP_PARAMS,
                'using_dp': [True, True, True, True]
            },
        ]
        method = 'step'
        seed = 89
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                print(f"Starting experiment with args: {exp}")
                runner.run_experiment(protocol=protocol,
                               eval_method=method, seed=seed, **exp)