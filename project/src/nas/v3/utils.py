import os
import shutil
import numpy as np
import pandas as pd

from itertools import groupby
from matplotlib import pyplot as plt
from typing import List, Dict


########################################################
#                       LOGGING                        #
########################################################


def clean_log():
    if not os.path.exists('LOGS'):
        os.mkdir('LOGS')
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            os.remove('LOGS/{}'.format(file))


def log_event():
    dest = 'LOGS'
    while os.path.exists(dest):
        dest = 'LOGS/event{}'.format(np.random.randint(10000))
    os.mkdir(dest)
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            shutil.move('LOGS/{}'.format(file),dest)


def get_latest_event_id():
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS') if os.path.isdir('LOGS/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/event', ''))


########################################################
#                 RESULTS PROCESSING                   #
########################################################


def load_nas_data():
    data = pd.read_csv('LOGS/event{}/nas_data.csv'.format(get_latest_event_id()))
    return data


########################################################
#                EVALUATION AND PLOTS                  #
########################################################


def get_top_n_architectures(data, top_n=5) -> List[Dict]:
    data_df = data.sort_values('reward', ascending=False, ignore_index=False)
    data_df = data_df.iloc[:top_n,:]
    print('Top {} Architectures:'.format(top_n))
    best_archs = []
    for idx,(arch,val_acc) in data_df.iterrows():
        print(f' . Architecture {idx}: {arch} | Validation accuracy: {val_acc}%')
        best_archs.append({'Architecture': arch, 'Validation accuracy': val_acc})
    return best_archs


def get_nas_accuracy_plot():
    data = load_nas_data()
    accuracies = [x for x in data.reward.values]
    plt.plot(np.arange(len(data)), accuracies)
    plt.show()


def get_accuracy_distribution():
    data = load_nas_data()
    accuracies = [x*100. for x in data.reward.values]
    accuracies = [int(x) for x in accuracies]
    sorted_accs = np.sort(accuracies)
    count_dict = {k: len(list(v)) for k, v in groupby(sorted_accs)}
    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.show()