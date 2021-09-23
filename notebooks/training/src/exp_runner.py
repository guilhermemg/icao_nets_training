import os
import sys
import yaml
import pprint
import argparse

import neptune.new as neptune

if '../../../../notebooks/' not in sys.path:
    sys.path.insert(0, '../../../../notebooks/')
    
from base.data_processor import DataProcessor
from base.model_trainer import ModelTrainer
from base.model_evaluator import ModelEvaluator, DataSource, DataPredSelection
from nas.nas_controller import NASController
from base.fake_data_producer import FakeDataProducer

if '..' not in sys.path:
    sys.path.insert(0, '..')

import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors
os.environ['NEPTUNE_API_TOKEN'] = cfg.NEPTUNE_API_TOKEN
os.environ['NEPTUNE_PROJECT'] = cfg.NEPTUNE_PROJECT


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
        self.nas_params = kwargs['nas_params']
        
        self.__kwargs_sanity_check()
        
        self.base_model = self.net_args['base_model']
        print('----')
        print('Base Model Name: ', self.base_model)
        print('----')
        
        self.neptune_run = None
        self.__start_neptune()
        
        print('----')
        self.is_mtl_model = len(self.prop_args['reqs']) > 1
        print(f'MTL Model: {self.is_mtl_model}')
        self.approach = self.prop_args['approach']
        print(f'Approach: {self.approach}')
        print('----')
        
        self.data_processor = DataProcessor(self.prop_args, self.net_args, self.is_mtl_model, self.neptune_run)
        self.nas_controller = NASController(self.nas_params, self.neptune_run, self.use_neptune)
        self.model_trainer = ModelTrainer(self.net_args, self.prop_args, self.base_model, self.is_mtl_model, self.approach, self.neptune_run)
        self.model_evaluator = ModelEvaluator(self.net_args, self.prop_args, self.is_mtl_model, self.neptune_run)
    
    
    def __kwargs_sanity_check(self):
        has_experiment_id = True if self.prop_args['orig_model_experiment_id'] != '' else False
        is_training_new_model = self.prop_args['train_model']
        
        if not has_experiment_id and not is_training_new_model:
            raise Exception('You must train a new model or provide an experiment ID')
        if has_experiment_id and is_training_new_model:
            raise Exception('You cannot train a new model and provide an experiment ID')
            
    
    def __start_neptune(self):
        self.__print_method_log_sig( 'start neptune')
        if self.use_neptune:
            print('Starting Neptune')
            self.neptune_run = neptune.init(name=self.exp_args['name'],
                                            description=self.exp_args['description'],
                                            tags=self.exp_args['tags'],
                                            source_files=self.exp_args['src_files'])    
        else:
            print('Not using Neptune to record Experiment Metadata')
    
    
    def __print_method_log_sig(self, msg):
        print(f'-------------------- {msg} -------------------')
    
    
    def __load_exp_config(self, yaml_config_file):
        self.__print_method_log_sig('load experiment configs')
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


    def produce_fake_data(self):
        self.__print_method_log_sig( 'producing fake data for experimental purposes')
        faker = FakeDataProducer(self.data_processor.train_data, 
                                 self.data_processor.validation_data,
                                 self.data_processor.test_data)
        faker.produce_data()

        self.data_processor.train_data = faker.fake_train_data_df
        self.data_processor.validation_data = faker.fake_validation_data_df
        self.data_processor.test_data = faker.fake_test_data_df

    
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

    
    def summary_labels_dist(self):
        self.__print_method_log_sig( 'summary labels dist')
        self.data_processor.summary_labels_dist()

    
    def summary_gen_labels_dist(self):
        self.__print_method_log_sig( 'summary gen labels dist')
        self.data_processor.summary_gen_labels_dist()    
    
       
    def setup_experiment(self):
        self.__print_method_log_sig( 'create experiment')
        if self.use_neptune:
            print('Setup neptune properties and parameters')

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
            props['orig_model_experiment_id'] = self.prop_args['orig_model_experiment_id']
            props['save_trained_model'] = self.prop_args['save_trained_model']
            props['sample_training_data'] = self.prop_args['sample_training_data']
            props['sample_prop'] = self.prop_args['sample_prop']
            props['is_mtl_model'] = self.is_mtl_model
            props['approach'] = self.prop_args['approach']
            
            self.neptune_run['parameters'] = params
            self.neptune_run['properties'] = props
            
            print('Properties and parameters setup done!')
        else:
            print('Not using Neptune')
    
    
    def __run_nas_trial(self, trial_num):
        print('+'*20 + ' STARTING NEW TRAIN ' + '+'*20)

        self.nas_controller.create_new_trial(trial_num)

        config = self.nas_controller.select_config()
        print(f' ----- Training {trial_num} | Config: {config} --------')
        
        vis_path = f'figs/nas/nas_model_{trial_num}.jpg'

        n_epochs = self.nas_params['n_epochs']

        self.model_trainer.create_model(config=config, running_nas=True)
        self.model_trainer.visualize_model(vis_path, verbose=False)
        self.model_trainer.train_model(self.train_gen, self.validation_gen, fine_tuned=False, n_epochs=n_epochs, running_nas=True)
        self.model_trainer.load_best_model()
        self.model_evaluator.set_data_src(DataSource.VALIDATION)
        self.model_evaluator.test_model(self.validation_gen, self.model_trainer.model, verbose=False, running_nas=True)

        self.nas_controller.evaluate_config(self.model_evaluator.req_evaluations)

        if self.use_neptune:
            self.neptune_run[f'viz/nas/model_architectures/nas_model_{trial_num}.jpg'].upload(vis_path)

        self.nas_controller.log_trial()
        self.nas_controller.finish_trial()

        print('-'*20 + 'FINISHING TRAIN' + '-'*20)


    def run_neural_architeture_search(self):
        self.__print_method_log_sig( 'run neural architecture search' )
        
        if self.use_neptune:
            self.neptune_run['nas_parameters'] = self.nas_params

        n_trials = self.nas_params['n_trials']
        for t in range(1,n_trials+1):
            self.__run_nas_trial(t)

        self.nas_controller.select_best_config()
        self.nas_controller.reset_memory()
        
    
    
    def create_model(self):
        self.__print_method_log_sig( 'create model')
        self.model_trainer.create_model(self.train_gen)
    
    
    def visualize_model(self, outfile_path):
        self.__print_method_log_sig( 'vizualize model')
        self.model_trainer.visualize_model(outfile_path)
    
    
    def train_model(self, fine_tuned=False, n_epochs=None):
        self.__print_method_log_sig( 'train model')
        self.model_trainer.train_model(self.train_gen, self.validation_gen, fine_tuned, n_epochs)
    
    
    def draw_training_history(self):
        self.__print_method_log_sig( 'draw training history')
        self.model_trainer.draw_training_history()
    
    
    def model_summary(self):
        self.model_trainer.model_summary()
    
    
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

    
    def set_model_evaluator_data_src(self, data_src):
        self.model_evaluator.set_data_src(data_src)
    
    
    def test_model(self):
        if self.model_evaluator.data_src.value == DataSource.TEST.value:
            self.model_evaluator.test_model(self.test_gen, self.model)
        if self.model_evaluator.data_src.value == DataSource.VALIDATION.value:
            self.model_evaluator.test_model(self.validation_gen, self.model)
            
    
    def vizualize_predictions(self, n_imgs=40, data_pred_selection=DataPredSelection.ANY):
        self.__print_method_log_sig( 'vizualize predictions')
        
        data_gen = None
        if self.model_evaluator.data_src.value == DataSource.TEST.value:
            data_gen = self.test_gen
        elif self.model_evaluator.data_src.value == DataSource.VALIDATION.value:
            data_gen = self.validation_gen
        
        self.model_evaluator.vizualize_predictions(base_model=self.base_model, 
                                                   model=self.model, 
                                                   data_gen=data_gen,
                                                   n_imgs=n_imgs, 
                                                   data_pred_selection=data_pred_selection)
    

    def finish_experiment(self):
        self.__print_method_log_sig( 'finish experiment')
        if self.use_neptune:
            print('Finishing Neptune')
            self.neptune_run.stop()
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
            self.setup_experiment()
            self.summary_labels_dist()
            self.summary_gen_labels_dist()
            self.create_model()
            self.vizualize_model(outfile_path=f"figs/model_architecture.png")
            self.train_model()
            self.draw_training_history()
            self.load_best_model()
            self.save_model()
            
            self.set_model_evaluator_data_src(DataSource.VALIDATION)
            self.test_model()
            self.vizualize_predictions(n_imgs=50)
            self.vizualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_TP)
            self.vizualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_FP)
            self.vizualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_FN)
            self.vizualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_TN)
            
            self.set_model_evaluator_data_src(DataSource.TEST)
            self.test_model()
            self.vizualize_predictions(n_imgs=50)
            self.vizualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_TP)
            self.vizualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_FP)
            self.vizualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_FN)
            self.vizualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_TN)
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