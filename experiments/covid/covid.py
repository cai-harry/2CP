import argparse

from experiments.runner import ExperimentRunner

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
        TRAINING_ITERATIONS = 3
    else:
        TRAINING_ITERATIONS = 16
    TRAINING_HYPERPARAMS = {
        'final_round_num': TRAINING_ITERATIONS,
        'batch_size': 8,
        'epochs': 16,
        'learning_rate': 1e-2
    }
    DP_PARAMS = {
        'l2_norm_clip': 0.7,
        'noise_multiplier': 2.0,
        'delta': 1e-5
    }

    ROUND_DURATION = 1200  # should always end early

    runner = ExperimentRunner(
        QUICK_RUN,
        TRAINING_ITERATIONS,
        TRAINING_HYPERPARAMS,
        ROUND_DURATION
    )

    if QUICK_RUN:
        experiments = [
            {
                'dataset': 'covid',
                'split_type': 'dp',
                'num_trainers': 4,
                'dp_params': DP_PARAMS,
                'using_dp': [True, True, True, True]
            },
            {
                'dataset': 'covid',
                'split_type': 'dp',
                'num_trainers': 4,
                'dp_params': DP_PARAMS,
                'using_dp': [False, True, True, True]
            },
            {
                'dataset': 'covid',
                'split_type': 'dp',
                'num_trainers': 4,
                'dp_params': DP_PARAMS,
                'using_dp': [True, False, False, False]
            },
        ]
        method = 'step'
        seed = 88
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                runner.run_experiment(protocol=protocol,
                                      eval_method=method, seed=seed, **exp)
    else:
        experiments = [
            {'dataset': 'covid', 'split_type': 'noniid',
                'num_trainers': 4, 'disjointness': 0.2},
            {'dataset': 'covid', 'split_type': 'noniid',
                'num_trainers': 4, 'disjointness': 0.4},
            {'dataset': 'covid', 'split_type': 'noniid',
                'num_trainers': 4, 'disjointness': 0.6},
            {'dataset': 'covid', 'split_type': 'noniid',
                'num_trainers': 4, 'disjointness': 0.8},
        ]
        method = 'step'
        seed = 89
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                print(f"Starting experiment with args: {exp}")
                runner.run_experiment(protocol=protocol,
                                      eval_method=method, seed=seed, **exp)
