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


def counts(results, percent=True):
    for r in results:
        n = r['num_trainers']
        names = NAMES[:n]
        round_idxs = range(0, r['final_round_num']+1)
        total_count = r['total_token_counts'][-1]
        if r['split_type'] == 'equal':
            details = [''] * len(names)
        if r['split_type'] == 'size':
            details = [f" (size={ratio})" for ratio in r['ratios']]
        if r['split_type'] == 'flip':
            details = [f" (flip={prob})" for prob in r['flip_probs']]
        for name, detail in zip(names, details):
            counts = np.array(r['token_counts'][name])
            counts = np.insert(counts, 0, 0)  # insert a 0 at the start of counts, as all trainers start with 0 tokens
            if percent:
                counts = 100 * (counts / total_count)
            plt.plot(
                round_idxs,
                counts,
                label=name + detail,
                marker='.')
        plt.xticks(round_idxs)
        plt.xlabel("Training round")
        if percent:
            plt.ylabel("Tokens (percent of total)")
        else:
            plt.ylabel("Tokens")
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
        path += f"-flip-prob-"
        path += '-'.join([str(p) for p in r['flip_probs']])
    if r['split_type'] == 'unique_digits':
        path += "-unique-"
        path += f"{r['num_trainers']}-trainers-"
        path += '-'.join([str(k) for k in r['unique_digits']])
    path += ".png"
    return path


if __name__ == "__main__":
    try:
        r = load_results({
            'final_round_num': 5,
            'eval_method': 'step',
        })
        counts(r)


    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit()
