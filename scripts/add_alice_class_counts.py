import json

DATASET = 'mnist'
# DATASET = 'covid'

RESULTS_FILE = {
    'mnist': "experiments/mnist/results/all.json",
    'covid': "experiments/covid/results/all.json",
}
KEY = {
    'mnist': 'digit_counts',
    'covid': 'label_counts'
}
ALICE_CLASS_COUNTS = {
    'mnist': [980, 1135, 1032, 1010,  982,  892,  958, 1028,  974, 1009],
    'covid': [79, 64],
}

with open(RESULTS_FILE[DATASET]) as f:
    results = json.load(f)

for r in results:
    if KEY[DATASET] in r:
        r[KEY[DATASET]]['Alice'] = ALICE_CLASS_COUNTS[DATASET]

with open(RESULTS_FILE[DATASET], 'w') as f:
    json.dump(results, f,
              indent=4,
              sort_keys=True)
