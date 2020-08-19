import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_FILE = {
    'mnist': "experiments/mnist/results/all.json",
    'covid': "experiments/covid/results/all.json",
}
PLOTS_DIR = {
    'mnist': "experiments/mnist/results/plots/",
    'covid': "experiments/covid/results/plots/",
}
NAMES = ["Alice", "Bob", "Carol", "David", "Eve",
         "Frank", "Georgia", "Henry", "Isabel", "Joe"]


def load_results(filters):
    # TODO: duplicated
    assert 'dataset' in filters, "Must specify dataset"
    with open(RESULTS_FILE[filters['dataset']]) as f:
        results = json.load(f)
    for key, value in filters.items():
        results = [r for r in results if r[key] == value]
    return results

def scatter(results, trainers):
    if not trainers:
        names_to_plot = NAMES[:1]
        filename = "gas-alice.png"
    else:
        names_to_plot = NAMES[1:]
        filename = "gas-trainers.png"
    for name in names_to_plot:
        x = []
        y = []
        for r in results:
            if 'gas_used' not in r or name not in r['gas_used']:
                continue
            x.append(r['num_trainers'])
            y.append(r['gas_used'][name])
        plt.scatter(x, y, label=name)
    plt.legend()
    plt.title(r['protocol'].capitalize())
    plt.xlabel("Number of trainers")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.ylabel("Gas used")
    filepath = PLOTS_DIR[r['dataset']] + filename
    plt.savefig(filepath)
    plt.clf()

if __name__ == "__main__":
    results = load_results({
        'seed': 89,
        'dataset': 'covid',
        'protocol': 'crowdsource',
        'final_round_num': 16,
    })
    for r in results:
        if 'gas_used' in r:
            print(r['gas_used'])
    scatter(results, False)
    scatter(results, True)
