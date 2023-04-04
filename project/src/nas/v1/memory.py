from src.nas.v1.trial import Trial
from deprecated import deprecated


@deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
class Memory:

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def __init__(self):
        self.trials = []

    
    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def reset(self):
        self.trials = []


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def is_empty(self):
        return len(self.trials) == 0


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def get_trials(self):
        return self.trials
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def get_last_trial(self) -> Trial:
        return self.trials[-1]
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def add_trial(self, trial: Trial):
        self.trials.append(trial)
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def contains(self, conf):
        for t in self.trials:
            c = t.get_config() 
            if c['n_denses_0'] == conf['n_denses_0'] and \
               c['n_denses_1'] == conf['n_denses_1'] and \
               c['n_denses_2'] == conf['n_denses_2'] and \
               c['n_denses_3'] == conf['n_denses_3']:
                return True
        return False



    