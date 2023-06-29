from src.nas.v3.nas_algorithm import NASAlgorithm
from src.nas.v3.mlp_search_space import MLPSearchSpaceIndicator
from src.base.experiment.training.optimizers import Optimizer
from src.base.experiment.training.base_models import BaseModel

from src.base.experiment.dataset.dataset import Dataset
from src.m_utils.nas_mtl_approach import NAS_MTLApproach
from src.m_utils.mtl_approach import MTLApproach
from src.m_utils.stl_approach import STLApproach

DATASET = Dataset.MNIST
APPROACH = NAS_MTLApproach.APPROACH_3


kwargs = { 
    # 'use_neptune': False,
    # 'exp_params' : {
    #     'name': 'NAS experiment',
    #     'description': 'NAS with Approach 3 - Testing parametrization of n_convs',
    #     'tags': [f'{DATASET.value["name"]}', 'nas', 'nas_approach_3_v2', 'no_use_predictor'],
    #     'src_files': ["../src/**/*.py"]
    # },
    # 'properties': {
    #     'approach': APPROACH,
    #     'dataset': DATASET,
    #     'tasks': DATASET.value['tasks'],
    #     'balance_input_data': False,
    #     'train_model': True,
    #     'save_trained_model': True,
    #     'exec_nas': True,
    #     'orig_model_experiment_id': '',
    #     'sample_training_data': True,
    #     'sample_prop': 0.02
    # },
    # 'nas_params': {
    #     'architecture_training_epochs': 2,     # n_epochs for training proposed architecture
    #     'total_num_proposed_architectures': 10,
    #     'nas_algorithm': NASAlgorithm.RL,
    #     'nas_search_space': MLPSearchSpaceIndicator.SS_2
    # },
    'controller_params': {
        'controller_max_proposed_arch_len': 5,   
        'controller_classes': 8,    # == n_candidates in search_space
        'controller_lstm_dim': 100,
        'controller_optimizer': Optimizer.ADAM,
        'controller_learning_rate': 0.01,
        'controller_decay': 0.1,
        'controller_momentum': 0.0,
        'controller_use_predictor': False,
        'controller_loss_alpha': 0.9,
        'controller_training_epochs': 2,
        'controller_batch_size': 256
    },
    # 'mlp_params': {
    #     'mlp_base_model': BaseModel.MOBILENET_V2,
    #     'mlp_n_epochs': 50,
    #     'mlp_batch_size': 64,
    #     'mlp_early_stopping': 50,
    #     'mlp_optimizer': Optimizer.ADAMAX,
    #     'mlp_learning_rate': 1e-3,
    #     'mlp_decay': 0.0,
    #     'mlp_momentum': 0.0,
    #     'mlp_dropout': 0.3,
    #     'mlp_loss_function': 'sparse_categorical_crossentropy',
    #     'mlp_one_shot': False
    # }
}


