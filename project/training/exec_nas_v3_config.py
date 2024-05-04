import sys

if '..' not in sys.path:
    sys.path.insert(0, '..')


from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DLName
from src.base.gt_loaders.gt_names import GTName
from src.base.experiment.dataset.dataset import Dataset
from src.base.experiment.training.base_models import BaseModel
from src.base.experiment.training.optimizers import Optimizer

from src.m_utils.stl_approach import STLApproach
from src.m_utils.mtl_approach import MTLApproach
from src.m_utils.nas_mtl_approach import NAS_MTLApproach

from src.nas.v3.nas_algorithm import NASAlgorithm
from src.nas.v3.mlp_search_space import MLPSearchSpaceIndicator


DATASET = Dataset.CELEB_A
APPROACH = NAS_MTLApproach.APPROACH_3


kwargs = { 
    'use_neptune': True,
    'exp_params' : {
        'name': 'NAS experiment',
        'description': 'NAS with Approach 3 - Experimenting with VGG-16 as base model',
        'tags': [f'{DATASET.value["name"]}', 'nas', 'nas_approach_3_v3_fixed', 'final_experiments_2'],
        'src_files': ["../src/**/*.py"]
    },
    'properties': {
        'approach': APPROACH,
        'dataset': DATASET,
        'tasks': DATASET.value['tasks'],
        'balance_input_data': False,
        'train_model': True,
        'save_trained_model': True,
        'exec_nas': True,
        'orig_model_experiment_id': '',
        'sample_training_data': False,
        'sample_prop': 1,
    },
    'nas_params': {
        'architecture_training_epochs': 5,     # n_epochs for training proposed architecture
        'total_num_proposed_architectures': 30,
        'nas_algorithm': NASAlgorithm.EVOLUTION,
        'nas_search_space': MLPSearchSpaceIndicator.SS_2
    },
    'controller_params': {
        'controller_max_proposed_arch_len': 8,   # == sss = 5 / tss = 6 / max_len = 8 (n_denses+n_convs)
        'controller_classes': MLPSearchSpaceIndicator.SS_2.value['n_classes'],    # == n_candidates ==> sss = 8 / n_operations ==> tss = 5 / classes = 8 (n_denses+n_convs)
        'controller_lstm_dim': 100,
        'controller_optimizer': Optimizer.ADAM,
        'controller_learning_rate': 0.006,
        'controller_decay': 0.0,
        'controller_momentum': 0.0,
        'controller_use_predictor': False,
        'controller_loss_alpha': 0.3,  # 0.9, 0.6, 0.3
        'controller_training_epochs': 20,
        'controller_batch_size': 10
    },
    'mlp_params': {
        'mlp_base_model': BaseModel.VGG16,
        'mlp_n_epochs': 50,
        'mlp_batch_size': 32,
        'mlp_early_stopping': 50,
        'mlp_optimizer': Optimizer.ADAMAX,
        'mlp_learning_rate': 1e-3,
        'mlp_decay': 0.0,
        'mlp_momentum': 0.0,
        'mlp_dropout': 0.3,
        'mlp_loss_function': 'sparse_categorical_crossentropy',
        'mlp_one_shot': False
    }
}