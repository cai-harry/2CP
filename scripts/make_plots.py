import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

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
    """
    Load all results whose parameters match those given in filters.
    """
    assert 'dataset' in filters, "Must specify dataset"
    with open(RESULTS_FILE[filters['dataset']]) as f:
        results = json.load(f)
    for key, value in filters.items():
        results = [r for r in results if r[key] == value]
    return results


def pie(results):
    """
    Make pie charts for all given results.
    """
    for r in results:
        n = r['num_trainers']
        names = NAMES[:n]
        tokens = [
            r['token_counts'][name][-1]
            for name in names
        ]
        plt.pie(tokens, labels=names, autopct='%1.1f%%')
        plt.savefig(_make_filepath(r, 'pie'))
        plt.clf()


def distributions(results, split_type=None):
    """
    Make plots of class distributions for given results, with the given split_type. (None=>all)
    """
    for r in results:
        if split_type is not None and r['split_type'] != split_type:
            continue
        if r['dataset'] == 'mnist':
            key = 'digit_counts'
        if r['dataset'] == 'covid':
            key = 'label_counts'
        if key not in r:
            continue
        lcs = r[key]
        names = r['trainers']
        if r['protocol'] == 'crowdsource':
            names.insert(0, "Alice")
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            lc = np.array(lcs[name])
            legend_label = name
            if r['protocol'] == 'crowdsource':
                ent = scipy.stats.entropy(lcs["Alice"], lc)
                legend_label += f" (entropy={ent:.3f})"
            distribution = 100 * lc / sum(lc)
            classes = np.arange(len(lc))
            plt.bar(
                classes + (i-1)*width,
                distribution,
                width,
                color=_PLOT_COLOUR(name),
                label=legend_label
            )
        plt.title(f"Class distributions with disj={r['disjointness']}")
        plt.xlabel('Class label')
        plt.ylabel('Examples (%)')
        plt.xticks(classes + width/2, classes)
        plt.legend()
        plt.savefig(_make_filepath(r, 'distr'))
        plt.clf()


def counts(results, percent=True):
    """
    Make plots of token counts for all given results. Plot as percentages of total if percent=True
    """
    for r in results:
        n = r['num_trainers']
        names = NAMES[:n]
        round_idxs = range(0, r['final_round_num']+1)
        total_count = r['total_token_counts'][-1]
        if r['split_type'] in {'equal', 'noniid'}:
            details = [''] * len(names)
        if r['split_type'] == 'size':
            details = [f" (size={ratio})" for ratio in r['ratios']]
        if r['split_type'] == 'flip':
            details = [f" (flip={prob})" for prob in r['flip_probs']]
        if r['split_type'] == 'unique_digits':
            details = [" (others)"] * len(names)
            details[0] = [f" ({','.join(r['unique_digits'])})"]
        if r['split_type'] == 'dp':
            details = [f" (w/out DP)"] * len(names)
            for i in range(len(names)):
                if r['using_dp'][i]:
                    eps = r['epsilon'][i]
                    delta = r['delta']
                    details[i] = f" (w/ DP, eps={eps:.2f}, delta={delta})"

        for name, detail in zip(names, details):
            counts = np.array(r['token_counts'][name])
            # insert a 0 at the start of counts, as all trainers start with 0 tokens
            counts = np.insert(counts, 0, 0)
            if percent and total_count > 0:
                counts = 100 * (counts / total_count)
            plt.plot(
                round_idxs,
                counts,
                label=name + detail,
                marker='.',
                color=_PLOT_COLOUR(name)
            )
        plt.xticks(round_idxs)
        plt.xlabel("Training round")
        if percent:
            plt.ylabel("Tokens (percent of total)")
        else:
            plt.ylabel("Tokens")
        plt.title(r['protocol'].capitalize())
        plt.legend()
        plt.savefig(_make_filepath(r, 'counts'))
        plt.clf()


def gas_history(results):
    """
    Plot gas use in all given results, where available.
    """
    for r in results:
        if 'gas_history' not in r:
            continue
        protocol = r['protocol']
        num_rounds = r['final_round_num']
        for name in ["Alice", *r['trainers']]:
            x = range(1, num_rounds+1)
            d = r['gas_history'][name]
            if protocol == 'crowdsource':
                y = [d[str(i)] for i in x]
            if protocol == 'consortium':
                if name == "Alice":
                    y = [d['1']]*len(x)
                else:
                    y = [sum([
                        d[k][str(i)] for k in d.keys()
                    ]) for i in x]
            y.insert(0, 0)
            plt.plot(y, label=name, marker='+', color=_PLOT_COLOUR(name))
        plt.xlabel("Training round")
        plt.xticks(x)
        plt.ylabel("Gas used")
        plt.ylim(0, 2e7)
        plt.title(protocol.capitalize())
        plt.legend()
        plt.savefig(_make_filepath(r, 'gas'))
        plt.clf()


def _make_filepath(r, plot_type):
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
        for dp in r['using_dp']:
            if dp:
                path += "t"
            else:
                path += "f"
        if r['noise_multiplier'] != 1.1:  # experiments from before first draft used only this value
            path += f"-{r['noise_multiplier']}"
    if r['seed'] != 89:  # 89 is default seed for single runs
        path += f"-seed-{r['seed']}"
    path += ".png"
    return path


def _PLOT_COLOUR(name):
    if name == "Alice":
        return 'black'
    else:
        idx = NAMES.index(name)
        return f"C{idx}"


if __name__ == "__main__":
    try:
        for dataset in ['mnist', 'covid']:
            r = load_results({
                'seed': 89,
                'dataset': dataset,
            })
            counts(r)
            # gas_history(r)
            # distributions(r, 'noniid')

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit()
