
from abc import ABC, abstractclassmethod

from src.nas.v1.memory import Memory
from src.nas.v1.trial import Trial
from src.base.experiment.evaluation.model_evaluator import DataSource
from deprecated import deprecated


@deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
class GenNASController(ABC):

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def __init__(self, model_trainer, model_evaluator, config_interp, neptune_utils):
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        
        self.config_interp = config_interp
        self.neptune_run = neptune_utils.neptune_run
        
        self.memory = Memory()
        self.cur_trial = None

        self.MAX_BLOCKS_PER_BRANCH = self.config_interp.nas_params['max_blocks_per_branch']
        self.N_CHILD_EPOCHS = self.config_interp.nas_params['n_child_epochs']

        self.best_config = None

    
    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def create_new_trial(self, trial_num):
        return Trial(trial_num)


    @abstractclassmethod    
    def run_nas_trial(self, trial_num, train_gen, validation_gen):
        raise NotImplemented()

  
    @abstractclassmethod
    def select_config(self):
        raise NotImplemented()

    
    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def train_child_architecture(self, train_gen, validation_gen):
        trial_num = self.cur_trial.get_num()
        config = self.cur_trial.get_config()

        print(f'\n\n ------ Training {trial_num} | Config: {config} -----\n')
        
        vis_path = f'training/figs/nas/nas_model_{trial_num}.jpg'

        self.model_trainer.create_model(config=config, running_nas=True)
        self.model_trainer.visualize_model(vis_path, verbose=False)
        self.model_trainer.train_model(train_gen, validation_gen, fine_tuned=False, n_epochs=self.N_CHILD_EPOCHS, running_nas=True)
        self.model_trainer.load_best_model()
        self.model_evaluator.set_data_src(DataSource.VALIDATION)
        final_eval = self.model_evaluator.test_model(validation_gen, self.model_trainer.model, verbose=False, running_nas=True)
        
        if self.config_interp.use_neptune:
            self.neptune_run[f'viz/nas/model_architectures/nas_model_{trial_num}.jpg'].upload(vis_path)

        return final_eval


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def log_trial(self):
        self.cur_trial.log_neptune(self.neptune_run, self.config_interp.use_neptune)


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def set_config_eval(self, eval):
        self.cur_trial.set_result(eval)


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def finish_trial(self):
        self.memory.add_trial(self.cur_trial)
        self.cur_trial = None


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def log_best_trial(self, best_trial):
        print(f'\nbest_trial: {best_trial}')
        if self.config_interp.use_neptune:
            self.neptune_run['nas/best_trial/num'] = best_trial.get_num()
            self.neptune_run['nas/best_trial/config'] = best_trial.get_config()
            self.neptune_run['nas/best_trial/final_EER_mean'] = best_trial.get_result()['final_EER_mean']
            self.neptune_run['nas/best_trial/final_ACC'] = best_trial.get_result()['final_ACC']


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
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

    
    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def reset_memory(self):
        self.memory.reset()
