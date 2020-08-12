import json

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = {
    'mnist': "experiments/mnist/results/all.json",
    'covid': "experiments/covid/results/all.json",
}
PLOTS_DIR = {
    'mnist': "experiments/mnist/results/plots/",
    'covid': "experiments/covid/results/plots/",
}
NAMES = ["Bob", "Carol", "David", "Eve",
         "Frank", "Georgia", "Henry", "Isabel", "Joe"]


def load_results(filters):
    assert 'dataset' in filters, "Must specify dataset"
    with open(RESULTS_FILE[filters['dataset']]) as f:
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
        if r['split_type'] == 'unique_digits':
            details = [f" (others)"] * len(names)
            details[0] = [f" ({','.join(r['unique_digits'])})"]
        if r['split_type'] == 'noniid':
            details = [f" (disj={r['disjointness']})"] * len(names)
        if r['split_type'] == 'dp':
            details = [f" (eps={eps})" for eps in r['epsilon']]
            
        for name, detail in zip(names, details):
            counts = np.array(r['token_counts'][name])
            counts = np.insert(counts, 0, 0)  # insert a 0 at the start of counts, as all trainers start with 0 tokens
            if percent and total_count > 0:
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
        plt.title(r['protocol'].capitalize())
        plt.legend()
        plt.savefig(make_filepath(r, 'counts'))
        plt.clf()
        

def make_filepath(r, plot_type):
    path = PLOTS_DIR[r['dataset']]
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
    if r['split_type'] == 'noniid':
        path += "-noniid-"
        path += f"{r['disjointness']}"
    if r['split_type'] == 'dp':
        path += "-dp-"
        raise NotImplementedError()
    path += ".png"
    return path


if __name__ == "__main__":
    try:
        r = load_results({
            'seed': 89,
            'dataset': 'covid',
        })
        counts(r)


    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit()
