import random
import pickle
import pyglove as pg
import numpy as np
import pandas as pd


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
        self.nas_data_log_path = 'LOGS/nas_data.pkl'


    def _feedback(self, dna, reward):
        self.nas_history_data.append([dna,reward])
        
        print(f' ..nas_history_data: {self.nas_history_data}')
        print(f' ..len(self.nas_history_data): {len(self.nas_history_data)}')

        self.__log_data()

    
    def __log_data(self):
        print(f'logging nas_history_data...')
        print(f' ..self.nas_history_data: {self.nas_history_data}')
        print(f' ..nas_data_log_path: {self.nas_data_log_path}')
        # df = pd.DataFrame(data=self.nas_history_data, columns=['dna', 'reward'])
        # df.to_csv(self.nas_data_log_path, index=False)
        with open(self.nas_data_log_path, 'wb') as f:
           pickle.dump(self.nas_data_log_path, f)


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
        
        return pg.geno.DNA(new_arch)

    