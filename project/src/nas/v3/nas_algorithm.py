import time
import random
import pickle
import pyglove as pg


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
        self.data : list = []
        self.nas_controller = NASController_4(config_interp)
        self.nas_data_log = 'LOGS/nas_data.pkl'
    

    def _feedback(self, dna, reward):
        self.data.append([dna,reward])
        print(f' ..new data: {self.data}')
        print(f' ..len(self.data): {len(self.data)}')

        self.__log_data()

    
    def __log_data(self):
        print('logging data...')
        #print(' ..self.data: ', self.data)
        #with open(self.nas_data_log, 'wb') as f:
        #    pickle.dump(self.data, f)
    
    
    def _propose(self):
        if len(self.data) % 2 == 0:
            self.train_model_controller()
        
        pred = [random.random() for _ in range(4)]
        print(f' ..new pred: {pred}')
        converted_pred = self.__convert_pred(pred)
        print(f' ..converted pred: {converted_pred}')
        return pg.geno.DNA(converted_pred)

    
    def __convert_pred(self, pred):
        converted_pred = []
        for v in pred:
            if v < 0.2:
                converted_pred.append(0)
            elif 0.2 <= v < 0.4:
                converted_pred.append(1)
            elif 0.4 <= v < 0.6:
                converted_pred.append(2)
            elif 0.6 <= v < 0.8:
                converted_pred.append(3)
            elif 0.8 <= v < 1.0:
                converted_pred.append(4)
        return converted_pred     

    
    def train_model_controller(self):
        print(f'\ntraining model - len(self.data) {len(self.data)}\n')
        time.sleep(3)

        self.nas_controller.data = self.data



        