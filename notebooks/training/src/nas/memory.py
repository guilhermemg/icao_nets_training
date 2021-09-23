

class Memory:
    def __init__(self):
        self.trials = []


    def reset(self):
        self.trials = []


    def get_trials(self):
        return self.trials
    

    def add_trial(self, trial):
        self.trials.append(trial)
    


    