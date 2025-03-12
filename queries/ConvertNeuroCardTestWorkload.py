from os.path import exists

import pandas as pd
import pickle as pkl
import os

workloads = ['job-light','job-light-ranges']
tags = ['', '_previous']
os.makedirs('./NeuroCard', exist_ok=True)
for tag in tags:
    for workload in workloads:
        df = pd.read_csv(f'./{workload}.csv', header=None, escapechar='\\', delimiter='#')
        with open(f'./{workload}_={tag}.pkl', 'rb') as f:
            true_cards = pkl.load(f)
        df.iloc[:,-1] = true_cards
        df.to_csv(f'./NeuroCard/{workload}_={tag}.csv', index=False, header=False, escapechar='\\', sep='#')