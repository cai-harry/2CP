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
    dataset = results[0]['dataset']
    assert all([r['dataset'] == dataset for r in results])
    protocol = results[0]['protocol']
    assert all([r['protocol'] == protocol for r in results])
    num_rounds = results[0]['final_round_num']
    assert all([r['final_round_num'] == num_rounds for r in results])
    if not trainers:
        names_to_plot = NAMES[:1]
        filename = f"gas-{protocol}-alice.png"
    else:
        names_to_plot = NAMES[1:]
        filename = f"gas-{protocol}-trainers.png"
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
    plt.title(f"{dataset.upper()}, {protocol.capitalize()}, {num_rounds} rounds")
    plt.xlabel("Number of trainers")
    plt.xticks(range(max([r['num_trainers'] for r in results])))  # x axis in integers from 0 to max number of trainers in results
    plt.ylabel("Gas used")
    filepath = PLOTS_DIR[dataset] + filename
    plt.savefig(filepath)
    plt.clf()

if __name__ == "__main__":

    for protocol in ['crowdsource', 'consortium']:
        print(f"MNIST, {protocol}")

        results = load_results({
            'seed': 89,
            'dataset': 'mnist',
            'protocol': protocol,
            'final_round_num': 5
        })
        for r in results:
            if 'gas_used' in r:
                print(r['gas_used'])
        scatter(results, False)
        scatter(results, True)


    for protocol in ['crowdsource', 'consortium']:
        print(f"COVID, {protocol}")

        results = load_results({
            'seed': 89,
            'dataset': 'covid',
            'protocol': protocol,
            'split_type': 'equal',
            'final_round_num': 16
        })
        for r in results:
            if 'gas_used' in r:
                print(r['gas_used'])
        scatter(results, False)
        scatter(results, True)
    
