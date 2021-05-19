import os
import sys
import yaml
import pprint
import neptune
import argparse

if '../../../notebooks/' not in sys.path:
    sys.path.append('../../../notebooks/')

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors
os.environ['NEPTUNE_API_TOKEN'] = cfg.NEPTUNE_API_TOKEN


class ExperimentRunner:
    def __init__(self, yaml_config_file=None, **kwargs):
        self.__print_method_log_sig( 'Init ExperimentRunner')
        
        if yaml_config_file != None:
            kwargs = self.__load_exp_config(yaml_config_file)
        
        print('---------------------------')
        print('Parent Process ID:', os.getppid())
        print('Process ID:', os.getpid())
        print('---------------------------')
        
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
        
        self.base_model = self.net_args['base_model']
        print('----')
        print('Base Model Name: ', self.base_model)
        print('----')
        
        print('----')
        self.is_mtl_model = len(self.prop_args['reqs']) > 1
        print(f'MTL Model: {self.is_mtl_model}')
        print('----')
        
        self.data_processor = DataProcessor(self.prop_args, self.net_args, self.is_mtl_model, self.use_neptune)
        self.model_trainer = ModelTrainer(self.net_args, self.prop_args, self.base_model, self.is_mtl_model, self.use_neptune)
        self.model_evaluator = ModelEvaluator(self.net_args, self.prop_args, self.is_mtl_model, self.use_neptune)
        
    
    def __print_method_log_sig(self, msg):
        print(f'-------------------- {msg} -------------------')
    
    
    def __load_exp_config(self, yaml_config_file):
        self.self.__print_method_log_sig('load experiment configs')
        print(f'Loading experiment config from {yaml_config_file}')
        with open(yaml_config_file, 'r') as f:
            cnt = yaml.load(f, Loader=yaml.Loader)[0]
            print('..Experiment configs loaded with success!')
            return cnt
    
    
    def load_training_data(self):
        self.__print_method_log_sig( 'load training data')
        self.data_processor.load_training_data()
        self.train_data = self.data_processor.train_data
        self.test_data = self.data_processor.test_data
        
    
    def sample_training_data(self):
        self.__print_method_log_sig( 'sample training data')
        if self.prop_args['sample_training_data']:
            self.data_processor.sample_training_data(self.prop_args['sample_prop'])
            self.train_data = self.data_processor.train_data
        else:
            print('Not applying subsampling in training data!')
    
        
    def balance_input_data(self):
        self.__print_method_log_sig( 'balance input data')
        if self.prop_args['balance_input_data']:
            req_name = self.prop_args['reqs'][0].value
            self.data_processor.balance_input_data(req_name)
            self.train_data = self.data_processor.train_data
        else:
            print('Not balancing input_data')
        
    
    def setup_data_generators(self):
        self.__print_method_log_sig( 'setup data generators')
        self.data_processor.setup_data_generators(self.base_model)
        self.train_gen = self.data_processor.train_gen
        self.validation_gen = self.data_processor.validation_gen
        self.test_gen = self.data_processor.test_gen

    
    def start_neptune(self):
        self.__print_method_log_sig( 'start neptune')
        if self.use_neptune:
            print('Starting Neptune')
            neptune.init('guilhermemg/icao-nets-training')    
        else:
            print('Not using Neptune')
        
    
    def summary_labels_dist(self):
        self.__print_method_log_sig( 'summary labels dist')
        self.data_processor.summary_labels_dist()

    
    def summary_gen_labels_dist(self):
        self.__print_method_log_sig( 'summary gen labels dist')
        self.data_processor.summary_gen_labels_dist()    
    
       
    def create_experiment(self):
        self.__print_method_log_sig( 'create experiment')
        if self.use_neptune:
            print('Creating experiment')

            params = self.net_args
            params['n_train'] = self.train_gen.n
            params['n_validation'] = self.validation_gen.n
            params['n_test'] = self.test_gen.n
            
            props = {}
            if self.prop_args['use_gt_data']:
                gt_names_formatted = {
                    'train_validation': [x.value.lower() for x in self.prop_args['gt_names']['train_validation']],
                    'test': [x.value.lower() for x in self.prop_args['gt_names']['test']],
                    'train_validation_test': [x.value.lower() for x in self.prop_args['gt_names']['train_validation_test']]
                }
                props = {'gt_names': str(gt_names_formatted)}
            else:
                props = {
                    'dl_names': str([dl_n.value for dl_n in self.prop_args['dl_names']]),
                    'tagger_model': self.prop_args['tagger_model'].get_model_name().value
                }

            props['aligned'] = self.prop_args['aligned']
            props['icao_reqs'] = str([r.value for r in self.prop_args['reqs']])
            props['balance_input_data'] = self.prop_args['balance_input_data']
            props['train_model'] = self.prop_args['train_model']
            props['model_name'] = self.prop_args['model_name']
            props['orig_model_experiment_id'] = self.prop_args['orig_model_experiment_id']
            props['save_trained_model'] = self.prop_args['save_trained_model']
            props['sample_training_data'] = self.prop_args['sample_training_data']
            props['sample_prop'] = self.prop_args['sample_prop']
            props['is_mtl_model'] = self.is_mtl_model
            
            neptune.create_experiment( name=self.exp_args['name'],
                                       params=params,
                                       properties=props,
                                       description=self.exp_args['description'],
                                       tags=self.exp_args['tags'],
                                       upload_source_files=self.exp_args['src_files'])
        else:
            print('Not using Neptune')
    
    
    def create_model(self):
        self.__print_method_log_sig( 'create model')
        self.model = self.model_trainer.create_model(self.train_gen)
    
    
    def vizualize_model(self, outfile_path):
        self.__print_method_log_sig( 'vizualize model')
        self.model_trainer.vizualize_model(outfile_path)
    
    
    def train_model(self):
        self.__print_method_log_sig( 'train model')
        self.model_trainer.train_model(self.train_gen, self.validation_gen)
    
    
    def draw_training_history(self):
        self.__print_method_log_sig( 'draw training history')
        self.model_trainer.draw_training_history()
    
    
    def load_checkpoint(self, chkp_name):
        self.__print_method_log_sig( 'load checkpoint')
        self.model_trainer.load_checkpoint(chkp_name)
        self.model = self.model_trainer.model
    

    def load_best_model(self):
        self.__print_method_log_sig( 'load best model')
        self.model_trainer.load_best_model()
        self.model = self.model_trainer.model
    
    
    def save_model(self):
        self.__print_method_log_sig( 'save model')
        if self.prop_args['save_trained_model']:
            self.model_trainer.save_trained_model()
        else:
            print('Not saving model!')

    
    def test_model(self):
        self.__print_method_log_sig( 'test model')
        self.model_evaluator.test_model(self.test_gen, self.model, self.is_mtl_model)

    
    def evaluate_model(self, data_src='test'):
        self.__print_method_log_sig( 'evaluate model')
        if data_src == 'validation':
            self.model_evaluator.evaluate_model(self.validation_gen, self.model)
        elif data_src == 'test':
            self.test_gen.reset()
            self.model_evaluator.evaluate_model(self.test_gen, self.model)
    
    
    def vizualize_predictions(self, n_imgs = 40, show_only_fp=False, show_only_fn=False, show_only_tp=False, show_only_tn=False):
        self.__print_method_log_sig( 'vizualize predictions')
        self.model_evaluator.vizualize_predictions(base_model=self.base_model, model=self.model, test_gen=self.test_gen,
                                                  n_imgs=n_imgs, 
                                                  show_only_fp=show_only_fp, show_only_fn=show_only_fn,
                                                  show_only_tp=show_only_tp, show_only_tn=show_only_tn)
    

    def finish_experiment(self):
        self.__print_method_log_sig( 'finish experiment')
        if self.use_neptune:
            print('Finishing Neptune')
            neptune.stop()
            self.use_neptune = False
        else:
            print('Not using Neptune')

        
    def run(self):
        self.__print_method_log_sig( 'run experiment')
        self.load_training_data()
        self.sample_training_data()
        self.balance_input_data()
        self.setup_data_generators()
        try:
            self.start_neptune()
            self.create_experiment()
            self.summary_labels_dist()
            self.summary_gen_labels_dist()
            self.create_model()
            self.train_model()
            self.draw_training_history()
            self.load_best_model()
            self.save_model()
            self.test_model()
            self.vizualize_predictions(n_imgs=50)
            self.vizualize_predictions(n_imgs=50, show_only_tp=True)
            self.vizualize_predictions(n_imgs=50, show_only_fp=True)
            self.vizualize_predictions(n_imgs=50, show_only_fn=True)
            self.vizualize_predictions(n_imgs=50, show_only_tn=True)
            return 0
        except Exception as e:
            print(f'ERROR: {e}')
            return 1
        finally:
            self.finish_experiment()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, dest='config_file', help='Path to yaml config file')
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config_file)
    runner.run()