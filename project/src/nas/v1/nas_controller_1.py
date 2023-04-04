
import numpy as np

from src.m_utils.constants import SEED
from src.nas.v1.gen_nas_controller import GenNASController
from deprecated import deprecated


@deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
class NASController_1(GenNASController):
    def __init__(self, model_trainer, model_evaluator, config_interp, neptune_utils):
        super().__init__(model_trainer, model_evaluator, config_interp, neptune_utils)

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def __gen_new_seed(self, x):
        return (self.cur_trial.get_num() * SEED) + SEED + x

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def select_config(self):
        i = 0
        rng = np.random.default_rng(self.__gen_new_seed(x=i))
        config = {f'n_denses_{i}':x for i,x in enumerate(rng.integers(low=1, high=self.MAX_BLOCKS_PER_BRANCH+1, size=4))}
        while(self.memory.contains(config)):
            print(' -- repeated config : selecting new one')
            rng = np.random.default_rng(self.__gen_new_seed(x=i))
            config = {f'n_denses_{i}':x for i,x in enumerate(rng.integers(low=1, high=self.MAX_BLOCKS_PER_BRANCH+1, size=4))}
            i += 1
        return config
    
    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
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
