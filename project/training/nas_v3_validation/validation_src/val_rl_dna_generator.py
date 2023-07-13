import random
import pyglove as pg
import numpy as np
import pandas as pd

from validation_src.val_config_interp import ConfigInterpreter
from validation_src.val_nas_controller import NASController_4


class RL_DNAGenerator(pg.generators.geno.DNAGenerator):
    def __init__(self, config_interp: ConfigInterpreter):
        super().__init__()
        self.nas_history_data : list = []
        self.nas_controller: NASController_4 = NASController_4(config_interp)
        self.nas_data_log_path = 'LOGS/nas_data.csv'


    def _feedback(self, dna, reward):
        self.nas_history_data.append([dna,reward])
        
        #print(f' ..nas_history_data: {self.nas_history_data}')
        #print(f' ..len(self.nas_history_data): {len(self.nas_history_data)}')

        self.__log_data()

    
    def __log_data(self):
        # print(f' ..Logging nas_history_data...')
        # print(f' ..self.nas_history_data: {self.nas_history_data}')
        # print(f' ..nas_data_log_path: {self.nas_data_log_path}')
        df = pd.DataFrame(data=self.nas_history_data, columns=['dna', 'reward'])
        df.to_csv(self.nas_data_log_path, index=False)
        # print(' ..done!')


    def _propose(self):
        if self.num_feedbacks % self.nas_controller.controller_batch_size == 0 and self.num_feedbacks > 0:
            # print(70*'.')
            # print(' ..New batch of architectures. Training controller model...')
            self.nas_controller.train_model_controller(self.nas_history_data)    
            # print(70*'.')
        
        prev_arch = None
        if self.num_feedbacks > 0:
            prev_arch = self.nas_history_data[-1][0].to_numbers()
        else:
            prev_arch = [random.randint(0,10) for _ in range(self.nas_controller.controller_max_proposed_arch_len)]

        prev_arch = np.array(prev_arch).reshape(1,1,self.nas_controller.controller_max_proposed_arch_len)

        # print(f' ..prev_arch: {prev_arch}')
        # print(f' ..prev_arch.shape: {prev_arch.shape}')

        new_arch = self.nas_controller.build_new_arch(prev_arch)
        
        return pg.geno.DNA(new_arch)