


class Trial:
    def __init__(self, num):
        self.num = num
        self.config = None
        self.result = None
    

    def get_num(self):
        return self.num


    def set_config(self, config):
        self.config = config
    

    def get_config(self):
        return self.config
    

    def set_result(self, result):
        self.result = result
    

    def get_result(self):
        return self.result
    

    def log_neptune(self, neptune_run, use_neptune):
        if use_neptune:
            neptune_run[f'nas/trial/{self.num}/config'] = self.get_config()
            neptune_run[f'nas/trial/{self.num}/result/final_EER_mean'] = self.get_result()['final_EER_mean']
            neptune_run[f'nas/trial/{self.num}/result/final_ACC'] = self.get_result()['final_ACC']
    

    def __str__(self):
        return f'Trial: \n  num: {self.num}\n  config: {self.config}\n  result: {self.result}'