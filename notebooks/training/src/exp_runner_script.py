import os
import sys
import pprint

if '../../../../notebooks/' not in sys.path:
    sys.path.insert(0, '../../../../notebooks/')

import utils.constants as cts

if '.' not in sys.path:
    sys.path.insert(0, '.')

from data_loaders.data_loader import DLName
from gt_loaders.gt_names import GTName
from exp_runner import ExperimentRunner
from model_trainer import BaseModel, Optimizer


def create_config(req, ds, aligned):
    return { 
                'use_neptune': True,
                'exp_params' : {
                    'name': 'train_vgg16',
                    'description': f'Training network for {req.value.upper()} requisite.',
                    'tags': ['vgg16', 'ground truths', 'adamax', ds.value.lower(), 'binary_output', req.value.lower()],
                    'src_files': ['src/*.py']
                },
                'properties': {
                    'reqs': [req],
                    'aligned': aligned,
                    'use_gt_data': True,
                    'gt_names': {
                        'train_validation': [],
                        'test': [],
                        'train_validation_test': [ds]
                    },
                    'balance_input_data': False,
                    'train_model': True,
                    'save_trained_model': True,
                    'orig_model_experiment_id': '',
                    'sample_training_data': False,
                    'sample_prop': 1.
                },
                'net_train_params': {
                    'base_model': BaseModel.VGG16,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'early_stopping': 10,
                    'learning_rate': 1e-3,
                    'optimizer': Optimizer.ADAMAX,
                    'dropout': 0.3
                }
            }


def run_experiment(l, cfgs):
    l.acquire()
    try:
        runner = ExperimentRunner(**cfgs)
        runner.run()
    finally:
        l.release()


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    #if os.path.exists('exp_logs/single_task_logs.log'):
    #    os.remove('exp_logs/single_task_logs.log')
    
    reqs_list = [cts.ICAO_REQ.RED_EYES, 
                 cts.ICAO_REQ.BACKGROUND, 
                 cts.ICAO_REQ.HAIR_EYES,
                 cts.ICAO_REQ.PIXELATION,
                 cts.ICAO_REQ.WASHED_OUT,
                 cts.ICAO_REQ.SKIN_TONE,
                 cts.ICAO_REQ.BLURRED,
                 cts.ICAO_REQ.INK_MARK]
    ds_list = [GTName.FVC]
    align_list = [False]
    
    lock = mp.Lock()
    for req in reqs_list:
        for ds in ds_list:
            for alig in align_list:
                exp_cf = create_config(req, ds, alig)
                p = mp.Process(target=run_experiment, args=(lock, exp_cf))
                p.start()
                p.join()
