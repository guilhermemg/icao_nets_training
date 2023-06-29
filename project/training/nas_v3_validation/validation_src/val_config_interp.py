import yaml
import pprint

from typing import List

from src.m_utils.nas_mtl_approach import NAS_MTLApproach
from src.m_utils.mtl_approach import MTLApproach
from src.m_utils.stl_approach import STLApproach
from src.m_utils.utils import print_method_log_sig
from src.base.experiment.tasks.task import TASK
from src.base.experiment.dataset.dataset import Dataset


class ConfigInterpreter:
    def __init__(self, kwargs, yaml_config_file=None):
        if yaml_config_file != None:
            kwargs = self.__load_exp_config(yaml_config_file)

        # self.use_neptune = kwargs['use_neptune']
        # print('-----')
        # print('Use Neptune: ', self.use_neptune)
        # print('-----')
        
        # print('-------------------')
        # print('Args: ')
        # pprint.pprint(kwargs)
        # print('-------------------')
        
        # self.exp_args = kwargs['exp_params']
        # self.prop_args = kwargs['properties']
        
        # self.mlp_params = kwargs['mlp_params']

        # self.nas_params = kwargs['nas_params']
        self.controller_params = kwargs['controller_params']
        
        # self.__kwargs_sanity_check()

        # self.dataset : Dataset = self.prop_args['dataset']
        # self.tasks : List[TASK] = self.prop_args['tasks']

        # self.base_model = self.mlp_params['mlp_base_model']
        # print('----')
        # print('Base Model Name: ', self.base_model)
        # print('----')

        # self.is_mtl_model = self.__get_is_mtl_model()
        # self.approach = self.__get_approach()
        # self.is_nas_mtl_model = self.__get_is_nas_mtl_model()
        # self.exec_nas = self.prop_args['exec_nas']
        
        # print('----')


    # def __get_approach(self):
    #     approach = self.prop_args['approach']

    #     if len(self.tasks) == 1:
    #         assert isinstance(approach, STLApproach)
    #     elif len(self.tasks) > 1:
    #         assert isinstance(approach, MTLApproach) or isinstance(approach, NAS_MTLApproach)
    #     else:
    #         raise Exception('Invalid approach')

    #     # print(f'Approach: {approach}')

    #     return approach
    

    # def __get_is_mtl_model(self):
    #     is_mtl_model = len(self.tasks) > 1
    #     # print(f'MTL Model: {is_mtl_model}')
    #     return is_mtl_model
    

    # def __get_is_nas_mtl_model(self):
    #     is_nas_mtl_model = len(self.tasks) > 1 and self.prop_args['exec_nas']
    #     # print(f'NAS MTL Model: {is_nas_mtl_model}')
    #     return is_nas_mtl_model


    def __load_exp_config(self, yaml_config_file):
        # print_method_log_sig('load experiment configs')
        # print(f'Loading experiment config from {yaml_config_file}')
        with open(yaml_config_file, 'r') as f:
            cnt = yaml.load(f, Loader=yaml.Loader)[0]
            # print('..Experiment configs loaded with success!')
            return cnt
    

    # def __kwargs_sanity_check(self):
    #     has_experiment_id = True if self.prop_args['orig_model_experiment_id'] != '' else False
    #     is_training_new_model = self.prop_args['train_model']
        
    #     if not has_experiment_id and not is_training_new_model:
    #         raise Exception('You must train a new model or provide an experiment ID')        
            

          