import yaml
import pprint

from src.m_utils.nas_mtl_approach import NAS_MTLApproach
from src.m_utils.utils import print_method_log_sig

class ConfigInterpreter:
    def __init__(self, kwargs, yaml_config_file=None):
        if yaml_config_file != None:
            kwargs = self.__load_exp_config(yaml_config_file)

        self.use_neptune = kwargs['use_neptune']
        print('-----')
        print('Use Neptune: ', self.use_neptune)
        print('-----')
        
        print('-------------------')
        print('Args: ')
        pprint.pprint(kwargs)
        print('-------------------')
        
        self.exp_args = kwargs['exp_params']
        self.prop_args = kwargs['properties']
        self.net_args = kwargs['net_train_params']
        self.nas_params = kwargs['nas_params']
        
        self.__kwargs_sanity_check()

        self.use_icao_gt = self.prop_args['icao_data']['icao_gt']['use_gt_data']
        self.use_icao_dl = self.prop_args['icao_data']['icao_dl']['use_dl_data']
        self.use_benchmark_data = self.prop_args['benchmarking']['use_benchmark_data']
        
        self.benchmark_dataset = self.prop_args['benchmarking']['benchmark_dataset']

        self.base_model = self.net_args['base_model']
        print('----')
        print('Base Model Name: ', self.base_model)
        print('----')

        self.is_mtl_model = self.__check_is_mtl_model()
        print(f'MTL Model: {self.is_mtl_model}')
        
        self.approach = self.prop_args['approach']
        print(f'Approach: {self.approach}')

        self.is_nas_mtl_model = type(self.approach) is NAS_MTLApproach
        self.exec_nas = self.prop_args['exec_nas']
        
        print('----')


    def __load_exp_config(self, yaml_config_file):
        print_method_log_sig('load experiment configs')
        print(f'Loading experiment config from {yaml_config_file}')
        with open(yaml_config_file, 'r') as f:
            cnt = yaml.load(f, Loader=yaml.Loader)[0]
            print('..Experiment configs loaded with success!')
            return cnt
    

    def __kwargs_sanity_check(self):
        has_experiment_id = True if self.prop_args['orig_model_experiment_id'] != '' else False
        is_training_new_model = self.prop_args['train_model']
        
        if not has_experiment_id and not is_training_new_model:
            raise Exception('You must train a new model or provide an experiment ID')


    def __check_is_mtl_model(self):
        if self.use_icao_gt:
            return len(self.prop_args['icao_data']['reqs']) > 1
        elif self.use_benchmark_data:
            return len(self.prop_args['benchmarking']['tasks']) > 1
        elif self.use_icao_dl:
            raise NotImplemented('MTL model is not implemented for ICAO DL!')