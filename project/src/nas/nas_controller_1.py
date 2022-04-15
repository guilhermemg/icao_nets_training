
import numpy as np

from src.m_utils.constants import SEED
from src.nas.gen_nas_controller import GenNASController


class NASController_1(GenNASController):
    def __init__(self, model_trainer, model_evaluator, config_interp, neptune_utils):
        super().__init__(model_trainer, model_evaluator, config_interp, neptune_utils)

    
    def __gen_new_seed(self, x):
        return (self.cur_trial.get_num() * SEED) + SEED + x


    def select_config(self):
        i = 0
        np.random.seed(self.__gen_new_seed(x=i))
        config = {f'n_denses_{i}':x for i,x in enumerate(np.random.randint(low=1, high=self.MAX_BLOCKS_PER_BRANCH+1, size=4))}
        while(self.memory.contains(config)):
            print(' -- repeated config : selecting new one')
            np.random.seed(self.__gen_new_seed(x=i))
            config = {f'n_denses_{i}':x for i,x in enumerate(np.random.randint(low=1, high=self.MAX_BLOCKS_PER_BRANCH+1, size=4))}
            i += 1
        return config
    

    def run_nas_trial(self, trial_num, train_gen, validation_gen):
        print('+'*20 + ' STARTING NEW TRAIN ' + '+'*20)

        self.cur_trial = self.create_new_trial(trial_num)
        config = self.select_config()
        self.cur_trial.set_config(config)    
        
        final_eval = self.train_child_architecture(train_gen, validation_gen)

        self.set_config_eval(final_eval)
        self.log_trial()
        self.finish_trial()

        print('-'*20 + 'FINISHING TRAIN' + '-'*20)
