
import numpy as np

from utils.constants import SEED

from nas.memory import Memory
from nas.trial import Trial

class NASController:
    def __init__(self, nas_params, neptune_run, use_neptune):
        self.nas_params = nas_params
        self.use_neptune = use_neptune
        self.neptune_run = neptune_run
        self.memory = Memory()
        self.cur_trial = None

        self.MAX_BLOCKS_PER_BRANCH = self.nas_params['max_blocks_per_branch']

        self.best_config = None


    def create_new_trial(self, trial_num):
        self.cur_trial = Trial(trial_num)

    
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


    def log_trial(self):
        self.cur_trial.log_neptune(self.neptune_run, self.use_neptune)


    def finish_trial(self):
        self.memory.add_trial(self.cur_trial)
        self.cur_trial = None


    def log_best_trial(self, best_trial):
        print(f'\nbest_trial: {best_trial}')
        if self.use_neptune:
            self.neptune_run['nas/best_trial/num'] = best_trial.get_num()
            self.neptune_run['nas/best_trial/config'] = best_trial.get_config()
            self.neptune_run['nas/best_trial/final_EER_mean'] = best_trial.get_result()['final_EER_mean']
            self.neptune_run['nas/best_trial/final_ACC'] = best_trial.get_result()['final_ACC']


    def select_best_config(self):
        trials = self.memory.get_trials()

        for t in trials:
            print(t)

        best_trial = None
        for trial in trials:
            if best_trial is None:
                best_trial = trial
            else:
                best_eer = best_trial.get_result()['final_EER_mean']
                cur_eer = trial.get_result()['final_EER_mean']
                if best_eer > cur_eer:
                    best_trial = trial

        self.log_best_trial(best_trial)

        self.best_config = best_trial.get_config()
        print(f'\nbest_config: {self.best_config}')

    
    def reset_memory(self):
        self.memory.reset()