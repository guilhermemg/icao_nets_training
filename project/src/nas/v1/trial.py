from deprecated import deprecated


@deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
class Trial:

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def __init__(self, num):
        self.num = num
        self.orig_vals = None  # float values of architecture/config before convertion to int (number of dense layers)
        self.config = None
        self.result = None
    
    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def get_num(self):
        return self.num


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def set_orig_vals(self, orig_vals):
        self.orig_vals = orig_vals


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def get_orig_vals(self):
        return self.orig_vals


    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def set_config(self, config):
        self.config = config
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def get_config(self):
        return self.config
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def set_result(self, result):
        self.result = result
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def get_result(self):
        return self.result
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def log_neptune(self, neptune_run, use_neptune):
        if use_neptune:
            neptune_run[f'nas/trial/{self.num}/config'] = self.get_config()
            neptune_run[f'nas/trial/{self.num}/orig_vals'] = self.get_orig_vals()
            neptune_run[f'nas/trial/{self.num}/result/final_EER_mean'] = self.get_result()['final_EER_mean']
            neptune_run[f'nas/trial/{self.num}/result/final_ACC'] = self.get_result()['final_ACC']
    

    @deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
    def __str__(self):
        return f'Trial: \n  num: {self.num}\n  config: {self.config}\n  orig_vals: {self.orig_vals}\n  result: {self.result}'