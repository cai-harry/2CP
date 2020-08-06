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
        TRAINING_ITERATIONS = 1
    else:
        TRAINING_ITERATIONS = 16
    TRAINING_HYPERPARAMS = {
        'final_round_num': TRAINING_ITERATIONS,
        'batch_size': 8,
        'epochs': 16,
        'learning_rate': 1e-2
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
            {'dataset': 'covid', 'split_type': 'flip', 'num_trainers': 4, 'flip_probs': [0.1,0,0,0]},
        ]
        method = 'step'
        seed = 88
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                runner.run_experiment(protocol=protocol,
                                      eval_method=method, seed=seed, **exp)
    else:
        experiments = [
            # {'dataset': 'covid', 'split_type': 'size', 'num_trainers': 6, 'ratios': [1,2,3,4,5,6]},
            {'dataset': 'covid', 'split_type': 'equal', 'num_trainers': 2},
            {'dataset': 'covid', 'split_type': 'equal', 'num_trainers': 3},
            {'dataset': 'covid', 'split_type': 'equal', 'num_trainers': 4},
            {'dataset': 'covid', 'split_type': 'equal', 'num_trainers': 5},
            {'dataset': 'covid', 'split_type': 'equal', 'num_trainers': 6},
            {'dataset': 'covid', 'split_type': 'flip', 'num_trainers': 4, 'flip_probs': [0,0,0,0.5]},
            {'dataset': 'covid', 'split_type': 'flip', 'num_trainers': 4, 'flip_probs': [0,0,0,0.4]},
            {'dataset': 'covid', 'split_type': 'flip', 'num_trainers': 4, 'flip_probs': [0,0,0,0.3]},
            {'dataset': 'covid', 'split_type': 'flip', 'num_trainers': 4, 'flip_probs': [0,0,0,0.2]},
            {'dataset': 'covid', 'split_type': 'flip', 'num_trainers': 4, 'flip_probs': [0,0,0,0.1]},
        ]
        method = 'step'
        seed = 89
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                print(f"Starting experiment with args: {exp}")
                runner.run_experiment(protocol=protocol,
                                      eval_method=method, seed=seed, **exp)
