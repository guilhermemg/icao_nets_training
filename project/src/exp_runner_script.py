import sys
import pandas as pd

if '.' not in sys.path:
    sys.path.insert(0, '.')

from src.m_utils import constants as cts
from src.base.gt_loaders.gt_names import GTName
from src.exp_runner import ExperimentRunner
from src.base.experiment.model_trainer import BaseModel, Optimizer


def get_experiment_id(req, aligned, ds):
    df = pd.read_csv('../analysis/single_task_nets/single_task_exps_data/icao-nets-training-2.csv')
    req_name = f"['{req.value.lower()}']"
    model_exp_id = df[(df['properties/icao_reqs'].str.contains(req_name, regex=False, case=False)) & 
                      (df['properties/aligned'] == float(aligned)) & 
                      (df['properties/gt_names'].str.contains(ds.value.lower()))].Id.values[0]
    return model_exp_id


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
                    'train_model': False,
                    'save_trained_model': True,
                    'orig_model_experiment_id': get_experiment_id(req, aligned, ds),
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
    
    reqs_list = list(cts.ICAO_REQ)
    ds_list = [GTName.FVC]
    align_list = [False]
    
    lock = mp.Lock()
    for req in reqs_list:
        if req.name == cts.ICAO_REQ.INK_MARK.name:
            pass
        for ds in ds_list:
            for alig in align_list:
                exp_cf = create_config(req, ds, alig)
                p = mp.Process(target=run_experiment, args=(lock, exp_cf))
                p.start()
                p.join()
