

class Memory:
    def __init__(self):
        self.trials = []


    def reset(self):
        self.trials = []


    def get_trials(self):
        return self.trials
    

    def add_trial(self, trial):
        self.trials.append(trial)
    

    def contains(self, conf):
        for t in self.trials:
            c = t.get_config() 
            if c['n_denses_1'] == conf['n_denses_1'] and c['n_denses_2'] == conf['n_denses_2'] and \
               c['n_denses_3'] == conf['n_denses_3'] and c['n_denses_4'] == conf['n_denses_4']:
                return True
        return False



    