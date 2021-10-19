
import numpy as np

from utils.constants import SEED
from nas.gen_nas_controller import GenNASController


class NASController_1(GenNASController):
    def __init__(self, nas_params, neptune_run, use_neptune):
        super().__init__(nas_params, neptune_run, use_neptune)

    
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
        self.cur_trial.set_config(config)
        return config
    

    def set_config_eval(self, eval):
        self.cur_trial.set_result(eval)
