import os
# disable tensorflow log level infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors

import sys
import pandas as pd

if '..' not in sys.path:
   sys.path.insert(0, '..')

from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DLName
from src.base.gt_loaders.gt_names import GTName
from src.exp_runner import ExperimentRunner

from src.base.experiment.dataset.dataset import Dataset
from src.base.experiment.evaluation.model_evaluator import DataSource, DataPredSelection
from src.base.experiment.training.base_models import BaseModel
from src.base.experiment.training.optimizers import Optimizer

from src.m_utils.mtl_approach import MTLApproach
from src.m_utils.nas_mtl_approach import NAS_MTLApproach


#N_TRIALS = 3
#NAS_APPROACH = NAS_MTLApproach.APPROACH_2
#NAS_APPROACH_STR = 'nas_approach_2'
#N_CHILD_EPOCHS = 1
#N_CHILD_EPOCHS_STR = '5_child_epochs'
#CONTROLLER_EPOCHS = 50
#N_EPOCHS = 3

#DATASET = Dataset.MNIST


kwargs = { 
    'use_neptune': False,
    'exp_params' : {
        'name': 'neural_arch_search',
        #'description': f'{NAS_APPROACH.value} with {DATASET.value["name"].upper()} dataset with {N_TRIALS} trials and patience and {N_CHILD_EPOCHS} child epoch',
        #'tags': ['nas', f'{NAS_APPROACH_STR}', 'benchmark', f'{DATASET.value["name"]}', f'{N_CHILD_EPOCHS_STR}'],
        'description': '',
        'tags': [],
        'src_files': ["../src/**/*.py"]
    },
    'properties': {
        'approach': NAS_MTLApproach.APPROACH_2,
        'benchmarking': {
            'use_benchmark_data': True,
            'dataset': Dataset.MNIST
        },
        'icao_data': {
            'icao_gt': {
                'use_gt_data': False,
                'gt_names': {
                    'train_validation': [],
                    'test': [],
                    'train_validation_test': [GTName.FVC]
                },
            },
            'icao_dl': {
                'use_dl_data': False,
                'tagger_model': None
            },
            'reqs': Dataset.FVC_ICAO.value['tasks'],
            'aligned': False
        },
        'balance_input_data': False,
        'train_model': True,
        'save_trained_model': True,
        'exec_nas': True,
        'orig_model_experiment_id': '',
        'sample_training_data': False,
        'sample_prop': 1.0
    },
    'nas_params': {
        #'max_blocks_per_branch': 5,
        #'n_child_epochs': N_CHILD_EPOCHS,
        #'controller_epochs': CONTROLLER_EPOCHS,
        #'controller_batch_size': 64,
        #'n_trials': N_TRIALS,

        'controller_sampling_epochs': 2,
        'samples_per_controller_epochs': 3,
        'controller_training_epochs': 5,
        'architecture_training_epochs': 2,
        'controller_loss_alpha': 0.9
    },
    'controller_params': {
        'controller_lstm_dim': 100,
        'controller_optimizer': 'Adam',
        'controller_learning_rate': 0.01,
        'controller_decay': 0.1,
        'controller_momentum': 0.0,
        'controller_use_predictor': False
    },
    'mlp_params': {
        'max_architecture_length': 5,
        'mlp_base_model': BaseModel.MOBILENET_V2,
        'mlp_n_epochs': 3,
        'mlp_batch_size': 64,
        'mlp_early_stopping': 5,
        'mlp_optimizer': Optimizer.ADAM,
        'mlp_learning_rate': 1e-2,
        'mlp_decay': 0.0,
        'mlp_momentum': 0.0,
        'mlp_dropout': 0.2,
        'mlp_loss_function': 'categorical_crossentropy',
        'mlp_one_shot': False
    }
}


runner = ExperimentRunner(**kwargs)

runner.load_training_data()
#runner.produce_fake_data()
runner.setup_data_generators()
runner.setup_experiment()
runner.run_neural_architecture_search_v2()
