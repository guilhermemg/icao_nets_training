import os
import sys
import pprint

if '../../../notebooks/' not in sys.path:
    sys.path.append('../../../notebooks/')

import utils.constants as cts

from data_loaders.data_loader import DLName
from gt_loaders.gt_names import GTName
from exp_runner import ExperimentRunner
from model_trainer import BaseModel, Optimizer


def create_config(req, ds, aligned):
    return { 
                'use_neptune': True,
                'script_mode': True,
                'exp_params' : {
                    'name': 'train_vgg16',
                    'description': f'Training network for {req.value.upper()} requisite',
                    'tags': ['vgg16', 'ground truths', 'adamax', ds.value.lower(), 'binary_output', req.value.lower()],
                    'src_files': ['exp_runner.py', 'data_processor.py', 'model_trainer.py', 'model_evaluator.py']
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
                    'model_name': '',
                    'sample_training_data': False,
                    'sample_prop': 1.
                },
                'net_train_params': {
                    'base_model': BaseModel.VGG16,
                    'batch_size': 64,
                    'n_epochs': 20,
                    'early_stopping': 10,
                    'learning_rate': 1e-3,
                    'optimizer': Optimizer.ADAMAX,
                    'train_prop': 0.9,
                    'validation_prop': 0.05,
                    'test_prop': 0.05,
                    'validation_split': 0.1,
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
    
    reqs_list = list(cts.ICAO_REQ)
    ds_list = [GTName.FVC, GTName.PYBOSSA]
    align_list = [True, False]
    
    lock = mp.Lock()
    for req in reqs_list:
        for ds in ds_list:
            for alig in align_list:
                exp_cf = create_config(req, ds, alig)
                p = mp.Process(target=run_experiment, args=(lock, exp_cf))
                p.start()
                p.join()
