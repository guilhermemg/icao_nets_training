import random
import pyglove as pg
import numpy as np
import pandas as pd

from validation_src.val_config_interp import ConfigInterpreter
from validation_src.val_nas_controller import NASController_4

from IPython.display import display

class RL_DNAGenerator(pg.generators.geno.DNAGenerator):
    def __init__(self, config_interp: ConfigInterpreter):
        super().__init__()
        self.nas_history_data : list = []
        self.nas_controller: NASController_4 = NASController_4(config_interp)
        self.nas_data_log_path = 'LOGS/nas_data.csv'


    def _feedback(self, dna, reward):
        #print(70*'*')
        #print(' ..Receiving feedback...')
        #print(f' ..dna: {dna} | reward: {reward}')

        #print(f'self.nas_history_data (feedback): {self.nas_history_data}')

        last_item = self.nas_history_data[-1]
        self.nas_history_data.pop()

        if self.nas_controller.controller_use_predictor:
            prev_dna = last_item[0]
            pred_acc = last_item[3]    
            self.nas_history_data.append([prev_dna,dna,reward,pred_acc])
            df = pd.DataFrame(data=self.nas_history_data, columns=['prev_dna','dna','reward','pred_acc'])
        else:
            prev_dna = last_item[0]
            self.nas_history_data.append([prev_dna,dna,reward])
            df = pd.DataFrame(data=self.nas_history_data, columns=['prev_dna','dna', 'reward'])

        # print(f' ..nas_history_data: {self.nas_history_data}')
        # print(f' ..len(self.nas_history_data): {len(self.nas_history_data)}')

        df.to_csv(self.nas_data_log_path, index=False)

        #print(70*'*')


    def _propose(self):
        #print(70*'*')
        #print(' ..Proposing new architecture...')

        if self.num_feedbacks % self.nas_controller.controller_batch_size == 0 and self.num_feedbacks > 0:
            # print(70*'.')
            # print(' ..New batch of architectures. Training controller model...')
            self.nas_controller.train_model_controller(self.nas_history_data)    
            # print(70*'.')
        
        #print(f'self.nas_history_data (propose): {self.nas_history_data}')

        prev_arch = None
        if self.num_feedbacks > 0:
            prev_arch = self.nas_history_data[-1][1].to_numbers()
        else:
            prev_arch = [random.randint(0,10) for _ in range(self.nas_controller.controller_max_proposed_arch_len)]

        #print(f' ..prev_arch (propose): {prev_arch}')

        prev_arch_dna = pg.geno.DNA(value=[[prev_arch]])

        prev_arch = np.array(prev_arch).reshape(1,1,self.nas_controller.controller_max_proposed_arch_len)        

        # print(f' ..prev_arch: {prev_arch}')
        # print(f' ..prev_arch.shape: {prev_arch.shape}')

        new_arch,pred_acc = self.nas_controller.build_new_arch(prev_arch)
        
        if self.nas_controller.controller_use_predictor:
            self.nas_history_data.append([prev_arch_dna,None,None,pred_acc])

        # print(f' ..new_arch: {new_arch} | pred_acc: {pred_acc}')
        # print(70*'*')

        return pg.geno.DNA(new_arch)