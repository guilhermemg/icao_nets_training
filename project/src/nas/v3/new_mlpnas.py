from datetime import datetime

import pandas as pd
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

        self.nas_data_df : pd.DataFrame = None
        self.nas_data_df_path: str = 'LOGS/full_nas_data_history.csv'
        self.sorted_nas_data_df_path: str = 'LOGS/sorted_nas_data_history.csv'

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


    def __run_nas(self):
        self.nas_data_df = pd.DataFrame(columns=['id','dna','reward','architecture','final_ACC','final_EER_mean','final_EER_median','final_EER_std_dv','elapsed_arch_training_time','elapsed_search_time'])

        init_total_search_time = datetime.now()

        for model,feedback in pg.sample(self.search_space, self.nas_algorithm, num_examples=self.total_num_proposed_architectures):
            init_arch_training_time = datetime.now()
        
            print(70*'=')
            print(f'  New Controller Epoch | Feedback ID: {feedback.id} | Feedback DNA: {feedback.dna}')
            print(70*'-')            
            
            model_spec = model()
            
            print(60*'-')
            print(f' -- Architecture {feedback.id}: {model_spec}')
            self.create_architecture(model_spec)
            self.train_architecture()
            final_eval = self.evaluate_architecture()
            print(60*'-')

            feedback(final_eval['final_ACC'])

            self.__log_nas_data(init_total_search_time, feedback, init_arch_training_time, model_spec, final_eval)

            print(70*'=')
        
        self.__sort_nas_log_data()
        
        self.nas_data_df.to_csv(self.sorted_nas_data_df_path, index=False)
        

    def __log_nas_data(self, init_total_search_time, feedback, init_arch_training_time, model_spec, final_eval):
        end_arch_training_time = datetime.now()
        elapsed_arch_training_time = (end_arch_training_time - init_arch_training_time).total_seconds()
        elapsed_search_time = (end_arch_training_time - init_total_search_time).total_seconds()

        self.nas_data_df = self.nas_data_df.append({'id': feedback.id,
                                                    'dna': feedback.dna,
                                                    'reward': final_eval['final_ACC'],
                                                    'architecture': model_spec, 
                                                    'final_ACC': final_eval['final_ACC'],
                                                    'final_EER_mean': final_eval['final_EER_mean'],
                                                    'final_EER_median': final_eval['final_EER_median'],
                                                    'final_EER_std_dv': final_eval['final_EER_std_dv'],
                                                    'elapsed_arch_training_time': elapsed_arch_training_time, 
                                                    'elapsed_search_time': elapsed_search_time,}, ignore_index=True)
        
        self.nas_data_df.to_csv(self.nas_data_df_path, index=False)


    def search(self):
        self.__run_nas()

        best_arch = self.__get_best_architecture()
        
        self.__print_nas_report()
        self.__log_nas_data_to_neptune()
        
        return best_arch
    

    def __get_best_architecture(self):
        return self.nas_data_df.at[0,'architecture']


    def __sort_nas_log_data(self):
        self.nas_data_df.sort_values('reward', ascending=False, ignore_index=False, inplace=True)


    def __print_nas_report(self):
        print('\n\n------------------ TOP ARCHITECTURES FOUND --------------------')
        for idx,row in self.nas_data_df.iloc[:5,:].iterrows():
            print(f' . Architecture {idx}: {row["architecture"]} | Validation accuracy: {row["final_ACC"]}%')
        print('------------------------------------------------------\n\n')


    def __log_nas_data_to_neptune(self):
        if self.config_interp.use_neptune:
            self.neptune_utils.log_nas_data(self.nas_data_df)
