import json

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = "experiments/mnist/results/all.json"
PLOTS_DIR = "experiments/mnist/results/plots/"
NAMES = ["Bob", "Carol", "David", "Eve",
         "Frank", "Georgia", "Henry", "Isabel", "Joe"]


def load_results(filters):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    for key, value in filters.items():
        results = [r for r in results if r[key] == value]
    return results


def pie(results):
    for r in results:
        n = r['num_trainers']
        names = NAMES[:n]
        tokens = [
            r['token_counts'][name][-1]
            for name in names
        ]
        plt.pie(tokens, labels=names, autopct='%1.1f%%')
        plt.savefig(make_filepath(r, 'pie'))
        plt.clf()


def counts(results):
    token_unit = 1e18
    for r in results:
        n = r['num_trainers']
        names = NAMES[:n]
        total_counts = np.array(r['total_token_counts']) / token_unit
        round_idxs = range(1, len(total_counts) + 1)
        for name in names:
            counts = np.array(r['token_counts'][name]) / token_unit
            plt.plot(
                round_idxs,
                counts,
                label=name,
                marker='.')
        plt.xticks(round_idxs)
        plt.ylim(0, max(total_counts))
        plt.xlabel("Training round")
        plt.ylabel("Tokens (1e18)")
        plt.legend()
        plt.savefig(make_filepath(r, 'counts'))
        plt.clf()


def shares(results):
    for r in results:
        n = r['num_trainers']
        names = NAMES[:n]
        total_counts = np.array(r['total_token_counts'])
        round_idxs = range(1, len(total_counts) + 1)
        for name in names:
            counts = np.array(r['token_counts'][name])
            shares = 100 * (counts / total_counts)
            plt.plot(
                round_idxs,
                shares,
                label=name,
                marker='.')
        plt.xticks(round_idxs)
        plt.ylim(0, 100)
        plt.xlabel("Training round")
        plt.ylabel("Share of model (%)")
        plt.legend()
        plt.savefig(make_filepath(r, 'shares'))
        plt.clf()

def make_filepath(r, plot_type):
    path = PLOTS_DIR
    path += f"{plot_type}-"
    path += r['protocol']
    if r['split_type'] == 'equal':
        path += "-equal"
        path += f"-{r['num_trainers']}-trainers"
    if r['split_type'] == 'size':
        path += "-sizes-"
        path += '-'.join([str(k) for k in r['ratios']])
    if r['split_type'] == 'flip':
        path += f"-flip-prob-{r['flip_probs'][0]}"
    if r['split_type'] == 'unique_digits':
        path += "-unique-"
        path += f"{r['num_trainers']}-trainers-"
        path += '-'.join([str(k) for k in r['unique_digits']])
    path += ".png"
    return path


if __name__ == "__main__":
    try:
        r = load_results({
            'seed': 88,
            'eval_method': 'step',
        })
        counts(r)


    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit()
