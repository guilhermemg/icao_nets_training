import numpy as np
import pyglove as pg

from tensorflow.keras.preprocessing.image import ImageDataGenerator


from src.nas.v3.mlp_search_space import New_MLPSearchSpace
from src.nas.v3.nas_algorithm import NASAlgorithmFactory
from src.nas.v3.utils import clean_log, log_event, load_nas_data

from src.configs.conf_interp import ConfigInterpreter
from src.m_utils.neptune_utils import NeptuneUtils

from src.base.experiment.training.model_trainer import ModelTrainer
from src.base.experiment.evaluation.model_evaluator import ModelEvaluator, DataSource


class New_MLPNAS:

    def __init__(self, train_gen, validation_gen, config_interp, neptune_utils):

        self.train_gen : ImageDataGenerator = train_gen
        self.validation_gen : ImageDataGenerator = validation_gen

        self.config_interp : ConfigInterpreter = config_interp
        self.neptune_utils : NeptuneUtils = neptune_utils

        self.total_num_proposed_architectures = self.config_interp.nas_params['total_num_proposed_architectures']
        self.architecture_train_epochs    = self.config_interp.nas_params['architecture_training_epochs']

        mlp_ss : New_MLPSearchSpace = New_MLPSearchSpace(self.config_interp.nas_params['nas_search_space'])
        self.search_space = mlp_ss.get_search_space()
        
        nas_algo_factory : NASAlgorithmFactory = NASAlgorithmFactory(self.config_interp.nas_params['nas_algorithm'])
        self.nas_algorithm = nas_algo_factory.get_algorithm(self.config_interp)

        self.data : list = []
        
        clean_log()

        self.model_trainer = ModelTrainer(config_interp, neptune_utils)
        self.model_evaluator = ModelEvaluator(config_interp, neptune_utils)


    def create_architecture(self, model_spec):
        self.model_trainer.create_model(self.train_gen, model_spec, running_nas=True)


    def train_architecture(self):
        self.model_trainer.train_model(self.train_gen, self.validation_gen, n_epochs=self.architecture_train_epochs, running_nas=True, fine_tuned=False)


    def evaluate_architecture(self):
        self.model_trainer.load_best_model()
        self.model_evaluator.set_data_src(DataSource.VALIDATION)
        final_eval = self.model_evaluator.test_model(self.validation_gen, self.model_trainer.model, verbose=False, running_nas=True)
        return final_eval

    
    def append_model_metrics(self, sequence, final_eval, pred_accuracy=None):
        if pred_accuracy:
            self.data.append([sequence, final_eval['final_ACC'], pred_accuracy])
        else:
            self.data.append([sequence, final_eval['final_ACC']])

        print('validation accuracy: ', final_eval['final_ACC'])


    def search(self):
        for model,feedback in pg.sample(self.search_space, self.nas_algorithm, num_examples=self.total_num_proposed_architectures):
        
            print(70*'=')
            print(f'  New Controller Epoch | Feedback ID: {feedback.id} | Feedback DNA: {feedback.dna}')
            print(70*'-')
                        
            #if self.use_predictor:
            #    pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
            #    print('Predicted accuracies: ', pred_accuracies)
            
            model_spec = model()
            
            print(f' -- Architecture {feedback.id}: {model_spec}')
            
            self.create_architecture(model_spec)
            self.train_architecture()
            final_eval = self.evaluate_architecture()
            
            # if self.use_predictor:
            #    self.append_model_metrics(model_spec, final_eval, pred_accuracies[i])
            # else:
            self.append_model_metrics(model_spec, final_eval)
            print(70*'=')

            feedback(final_eval['final_ACC'])
        
        log_event()

        self.__print_nas_report()
        self.__log_nas_data()
    

    def __print_nas_report(self):
        print('\n\n------------------ TOP ARCHITECTURES FOUND --------------------')
        nas_data_df = load_nas_data()
        nas_data_df = nas_data_df.sort_values('reward', ascending=False, ignore_index=False)
        for idx,(arch,val_acc) in nas_data_df.iloc[:5,:].iterrows():
             print(f' . Architecture {idx}: {arch} | Validation accuracy: {val_acc}%')
        print('------------------------------------------------------\n\n')
    

    def __log_nas_data(self):
        if self.config_interp.use_neptune:
            nas_data_df = load_nas_data()
            nas_data_df = nas_data_df.sort_values('reward', ascending=False, ignore_index=False)
            self.neptune_utils.log_nas_data(nas_data_df)