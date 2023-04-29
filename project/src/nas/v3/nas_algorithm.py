import time
import random
import pickle
import pyglove as pg
import numpy as np


from src.nas.v3.nas_controller_4 import NASController_4
from src.configs.conf_interp import ConfigInterpreter


class NASAlgorithmFactory(object):
    def __init__(self, name):
        self.name = name

    def get_algorithm(self, config_interp: ConfigInterpreter = None):
        """Creates algorithm."""
        if self.name == 'random':
            return pg.generators.Random()
        elif self.name == 'evolution':
            return pg.evolution.regularized_evolution(mutator=pg.evolution.mutators.Uniform(), population_size=50, tournament_size=10)
        elif self.name == 'rl':
            return RL_DNAGenerator(config_interp)
        else:
            return pg.load(self.name)


class RL_DNAGenerator(pg.generators.geno.DNAGenerator):
    def __init__(self, config_interp: ConfigInterpreter):
        self.nas_history_data : list = []
        self.nas_controller: NASController_4 = NASController_4(config_interp)
        self.nas_data_log = 'LOGS/nas_data.pkl'
        

    def _feedback(self, dna, reward):
        self.nas_history_data.append([dna,reward])
        
        print(f' ..nas_history_data: {self.nas_history_data}')
        print(f' ..len(self.nas_history_data): {len(self.nas_history_data)}')

        self.__log_data()

    
    def __log_data(self):
        print('logging data...')
        #print(' ..self.data: ', self.data)
        #with open(self.nas_data_log, 'wb') as f:
        #    pickle.dump(self.data, f)
    
    
    def _propose_BAK(self):
        if self.num_feedbacks % 2 == 0 and self.num_feedbacks > 0:
            self.nas_controller.train_model_controller(self.nas_history_data)    
        
        #seed = np.array([[[random.randint(0,50) for _ in range(4)]]])
        prev_arch = None
        if self.num_feedbacks > 0:
            prev_arch = self.nas_history_data[-1][0].to_numbers()
        else:
            prev_arch = [random.randint(0,5) for _ in range(4)]

        prev_arch = np.array([[prev_arch]])

        print(f' ..prev_arch: {prev_arch}')
        print(f' ..prev_arch.shape: {prev_arch.shape}')
        
        if self.nas_controller.use_predictor:
            new_arch,val_acc = self.nas_controller.controller_model.predict(prev_arch)
        else:
            new_arch = self.nas_controller.controller_model.predict(prev_arch)
        
        print(f' ..new arch: {new_arch}')
        print(f' ..new arch.shape: {new_arch.shape}')
        
        converted_arch = self.__convert_pred(new_arch)
        print(f' ..converted arch: {converted_arch}')
        
        return pg.geno.DNA(converted_arch)



    def _propose(self):
        samples_per_controller_epoch = self.nas_controller.samples_per_controller_epoch
        if self.num_feedbacks % samples_per_controller_epoch == 0 and self.num_feedbacks > 0:
            print(70*'.')
            print(' ..New batch of architectures. Training controller model...')
            self.nas_controller.train_model_controller(self.nas_history_data)    
            print(70*'.')
        
        prev_arch = None
        if self.num_feedbacks > 0:
            prev_arch = self.nas_history_data[-1][0].to_numbers()
        else:
            prev_arch = [random.randint(0,5) for _ in range(4)]

        prev_arch = np.array(prev_arch).reshape(1,1,4)

        print(f' ..prev_arch: {prev_arch}')
        print(f' ..prev_arch.shape: {prev_arch.shape}')

        new_arch = self.nas_controller.build_new_arch(prev_arch)
        
        #converted_arch = self.__convert_pred(new_arch)
        #print(f' ..converted arch: {converted_arch}')
        
        return pg.geno.DNA(new_arch)

    
    # def __convert_pred(self, pred):
    #     converted_pred = []
    #     for v in pred:
    #         if v < 0.2:
    #             converted_pred.append(0)
    #         elif 0.2 <= v < 0.4:
    #             converted_pred.append(1)
    #         elif 0.4 <= v < 0.6:
    #             converted_pred.append(2)
    #         elif 0.6 <= v < 0.8:
    #             converted_pred.append(3)
    #         elif 0.8 <= v < 1.0:
    #             converted_pred.append(4)
    #     return converted_pred        