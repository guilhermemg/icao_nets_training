import os
import copy
import argparse

from typing import List, Dict

from deprecated import deprecated

from src.base.experiment.data_loading.data_processor import DataProcessor
from src.base.experiment.training.model_trainer import ModelTrainer
from src.base.experiment.evaluation.model_evaluator import ModelEvaluator, DataSource, DataPredSelection
from src.base.experiment.data_loading.fake_data_producer import FakeDataProducer
from src.m_utils.neptune_utils import NeptuneUtils
from src.nas.v1.nas_controller_factory import NASControllerFactory
from src.nas.v2.mlpnas import MLPNAS
from src.nas.v3.new_mlpnas import New_MLPNAS
from src.m_utils.utils import print_method_log_sig
from src.configs.conf_interp import ConfigInterpreter
from src.configs import config as cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors
os.environ['NEPTUNE_API_TOKEN'] = cfg.NEPTUNE_API_TOKEN
os.environ['NEPTUNE_PROJECT'] = cfg.NEPTUNE_PROJECT


class ExperimentRunner:
    def __init__(self, yaml_config_file=None, **kwargs):
        print_method_log_sig( 'Init ExperimentRunner')
        
        print('---------------------------')
        print('Parent Process ID:', os.getppid())
        print('Process ID:', os.getpid())
        print('---------------------------')

        self.config_interp = ConfigInterpreter(kwargs, yaml_config_file)

        self.neptune_utils = NeptuneUtils(self.config_interp)

        self.data_processor = DataProcessor(self.config_interp, self.neptune_utils)
        self.model_trainer = ModelTrainer(self.config_interp, self.neptune_utils)
        self.model_evaluator = ModelEvaluator(self.config_interp, self.neptune_utils)

        if self.config_interp.use_neptune:
            self.neptune_utils.log_kwargs(kwargs)

    
    def load_training_data(self) -> None:
        print_method_log_sig( 'load training data')
        self.data_processor.load_training_data()
        self.train_data = self.data_processor.train_data
        self.test_data = self.data_processor.test_data


    def produce_fake_data(self) -> None:
        print_method_log_sig( 'producing fake data for experimental purposes')
        faker = FakeDataProducer(self.config_interp,
                                 self.data_processor.train_data, 
                                 self.data_processor.validation_data,
                                 self.data_processor.test_data)
        faker.produce_data()

        self.data_processor.train_data = faker.fake_train_data_df
        self.data_processor.validation_data = faker.fake_validation_data_df
        self.data_processor.test_data = faker.fake_test_data_df

    
    def sample_training_data(self) -> None:
        print_method_log_sig( 'sample training data')
        if self.config_interp.prop_args['sample_training_data']:
            self.data_processor.sample_training_data(self.config_interp.prop_args['sample_prop'])
            self.train_data = self.data_processor.train_data
        else:
            print('Not applying subsampling in training data!')
    
        
    def balance_input_data(self) -> None:
        print_method_log_sig( 'balance input data')
        if self.config_interp.prop_args['balance_input_data']:
            req_name = self.config_interp.prop_args['icao_gt']['reqs'][0].value
            self.data_processor.balance_input_data(req_name)
            self.train_data = self.data_processor.train_data
        else:
            print('Not balancing input_data')
        
    
    def setup_data_generators(self) -> None:
        print_method_log_sig( 'setup data generators')
        self.data_processor.setup_data_generators(self.config_interp.base_model)
        self.train_gen = self.data_processor.train_gen
        self.validation_gen = self.data_processor.validation_gen
        self.test_gen = self.data_processor.test_gen

    
    def summary_labels_dist(self) -> None:
        print_method_log_sig( 'summary labels dist')
        self.data_processor.summary_labels_dist()

    
    def summary_gen_labels_dist(self) -> None:
        print_method_log_sig( 'summary gen labels dist')
        self.data_processor.summary_gen_labels_dist()    
    
       
    def setup_experiment(self) -> None:
        print_method_log_sig( 'create experiment')
        if self.config_interp.use_neptune:
            print('Setting up neptune experiment')

            mlp_params = copy.deepcopy(self.config_interp.mlp_params)
            mlp_params['n_train'] = self.train_gen.n
            mlp_params['n_validation'] = self.validation_gen.n
            mlp_params['n_test'] = self.test_gen.n
            mlp_params['mlp_base_model'] = str(mlp_params['mlp_base_model'])
            mlp_params['mlp_optimizer'] = str(mlp_params['mlp_optimizer'])
            
            props = {}
            
            dataset = self.config_interp.dataset
            tasks = self.config_interp.tasks

            props['dataset'] = str(dataset.name)
            props['tasks'] = str([x.name for x in tasks])
            props['approach'] = str(self.config_interp.prop_args['approach'])
            props['balance_input_data'] = self.config_interp.prop_args['balance_input_data']
            props['train_model'] = self.config_interp.prop_args['train_model']
            props['orig_model_experiment_id'] = self.config_interp.prop_args['orig_model_experiment_id']
            props['save_trained_model'] = self.config_interp.prop_args['save_trained_model']
            props['sample_training_data'] = self.config_interp.prop_args['sample_training_data']
            props['sample_prop'] = self.config_interp.prop_args['sample_prop']
            props['is_mtl_model'] = self.config_interp.is_mtl_model
            
            contr_params = copy.deepcopy(self.config_interp.controller_params)
            contr_params['controller_optimizer'] = str(contr_params['controller_optimizer'])

            nas_params = copy.deepcopy(self.config_interp.nas_params)
            nas_params['nas_algorithm'] = str(nas_params['nas_algorithm'])
            nas_params['nas_search_space'] = str(nas_params['nas_search_space'])

            self.neptune_utils.neptune_run['properties'] = props
            self.neptune_utils.neptune_run['mlp_params'] = mlp_params
            self.neptune_utils.neptune_run['nas_params'] = nas_params
            self.neptune_utils.neptune_run['controller_params'] = contr_params            
            
            print('Neptune experiment setup done!')
        else:
            print('Not using Neptune')
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def run_neural_architecture_search(self) -> None:
        print_method_log_sig( 'run neural architecture search' )

        if not self.config_interp.exec_nas:
            print('Not executing neural architecture search')
            return

        self.nas_controller = NASControllerFactory.create_controller(self.config_interp, self.model_trainer, self.model_evaluator, self.neptune_utils)

        if self.config_interp.use_neptune:
            self.neptune_utils.neptune_run['nas_parameters'] = self.config_interp.nas_params

        if self.config_interp.exec_nas:
            print(f'Executing neural architectural search')
            self.nas_controller.reset_memory()
            print('  Memory reseted')
            
            for t in range(1,self.config_interp.nas_params['n_trials'] + 1):
                self.nas_controller.run_nas_trial(t, self.train_gen, self.validation_gen)
                        
            self.nas_controller.select_best_config()
        else:
            if self.config_interp.use_neptune:
                print(f'Not executing neural architecture search')
                self.neptune_utils.get_nas_data(self.config_interp.nas_params['n_trials'])
            else:
                print(f'Not executing neural architecture search and not using Neptune')
    

    def run_neural_architecture_search_v2(self) -> List[Dict]:
        print_method_log_sig( 'run neural architecture search' )

        if not self.config_interp.exec_nas:
            print('Not executing neural architecture search')
            return

        self.nas_object = MLPNAS(self.train_gen, self.validation_gen, self.config_interp, self.neptune_utils)
        self.nas_object.search()

        print('\n\n------------------ TOP ARCHITECTURES FOUND --------------------')
        best_archs_list = self.nas_object.get_top_n_architectures(5)
        print('----------------------------------------------------------------\n\n')

        if self.config_interp.use_neptune:
            self.neptune_utils.log_top_architectures_found(best_archs_list)
            self.neptune_utils.log_nas_data()

        return best_archs_list
    

    def run_neural_architecture_search_v3(self) -> List[Dict]:
        print_method_log_sig( 'run neural architecture search' )

        if not self.config_interp.exec_nas:
            print('Not executing neural architecture search')
            return

        self.nas_object = New_MLPNAS(self.train_gen, self.validation_gen, self.config_interp, self.neptune_utils)        
        best_arch = self.nas_object.search()

        return best_arch
    
    
    def create_model(self, config=None):
        print_method_log_sig( 'create model')
        #if self.config_interp.is_nas_mtl_model:
            #if self.config_interp.exec_nas:
            #    self.model_trainer.create_model(self.train_gen, config=config)
            #else:
        self.model_trainer.create_model(self.train_gen, config=config)
        #else:
        #    self.model_trainer.create_model(self.train_gen)
    
    
    def visualize_model(self, outfile_path):
        print_method_log_sig( 'vizualize model')
        self.model_trainer.visualize_model(outfile_path)
    
    
    def train_model(self, fine_tuned=False, n_epochs=None, running_nas=False):
        print_method_log_sig( 'train model')
        self.model_trainer.train_model(self.train_gen, self.validation_gen, fine_tuned, n_epochs, running_nas)
    
    
    def draw_training_history(self):
        print_method_log_sig( 'draw training history')
        self.model_trainer.draw_training_history()
    
    
    def model_summary(self):
        self.model_trainer.model_summary()
    
    
    def load_checkpoint(self, chkp_name):
        print_method_log_sig( 'load checkpoint')
        self.model_trainer.load_checkpoint(chkp_name)
        self.model = self.model_trainer.model
    

    def load_best_model(self):
        print_method_log_sig( 'load best model')
        self.model_trainer.load_best_model()
        self.model = self.model_trainer.model
    
    
    def save_model(self):
        print_method_log_sig( 'save model')
        if self.config_interp.prop_args['save_trained_model']:
            self.model_trainer.save_trained_model()
        else:
            print('Not saving model!')

    
    def set_model_evaluator_data_src(self, data_src):
        self.model_evaluator.set_data_src(data_src)
    
    
    def test_model(self, verbose=True):
        if self.model_evaluator.data_src.value == DataSource.TEST.value:
            self.model_evaluator.test_model(data_gen=self.test_gen, model=self.model, verbose=verbose)
        if self.model_evaluator.data_src.value == DataSource.VALIDATION.value:
            self.model_evaluator.test_model(data_gen=self.validation_gen, model=self.model, verbose=verbose)
            
    
    def visualize_predictions(self, n_imgs=40, data_pred_selection=DataPredSelection.ANY):
        print_method_log_sig( 'visualize predictions')
        
        data_gen = None
        if self.model_evaluator.data_src.value == DataSource.TEST.value:
            data_gen = self.test_gen
        elif self.model_evaluator.data_src.value == DataSource.VALIDATION.value:
            data_gen = self.validation_gen
        
        self.model_evaluator.visualize_predictions(base_model=self.config_interp.base_model, 
                                                   model=self.model, 
                                                   data_gen=data_gen,
                                                   n_imgs=n_imgs, 
                                                   data_pred_selection=data_pred_selection)
    

    def finish_experiment(self):
        print_method_log_sig( 'finish experiment')
        if self.config_interp.use_neptune:
            print('Finishing Neptune')
            self.neptune_utils.neptune_run.stop()
            self.config_interp.use_neptune = False
        else:
            print('Not using Neptune')

        
    def run(self):
        print_method_log_sig( 'run experiment')
        self.load_training_data()
        self.sample_training_data()
        self.balance_input_data()
        self.setup_data_generators()
        try:
            self.setup_experiment()
            self.summary_labels_dist()
            self.summary_gen_labels_dist()
            self.create_model()
            self.visualize_model(outfile_path=f"training/figs/model_architecture.png")
            self.train_model()
            self.draw_training_history()
            self.load_best_model()
            self.save_model()
            
            self.set_model_evaluator_data_src(DataSource.VALIDATION)
            self.test_model()
            self.visualize_predictions(n_imgs=50)
            self.visualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_TP)
            self.visualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_FP)
            self.visualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_FN)
            self.visualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_TN)
            
            self.set_model_evaluator_data_src(DataSource.TEST)
            self.test_model()
            self.visualize_predictions(n_imgs=50)
            self.visualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_TP)
            self.visualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_FP)
            self.visualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_FN)
            self.visualize_predictions(n_imgs=50, data_pred_selection=DataPredSelection.ONLY_TN)
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