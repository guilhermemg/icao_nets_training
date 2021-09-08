import numpy as np

from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint

from optimizers import Optimizer

class CallbacksHandler:
    def __init__(self, net_args, prop_args, use_neptune, neptune_run, checkpoint_path, is_mtl_model):
        self.net_args = net_args
        self.prop_args = prop_args
        self.use_neptune = use_neptune
        self.neptune_run = neptune_run
        self.checkpoint_path = checkpoint_path
        self.is_mtl_model = is_mtl_model


    def __log_data(self, logs):
        if not self.is_mtl_model:
            self.neptune_run['epoch/accuracy'].log(logs['accuracy'])
            self.neptune_run['epoch/val_accuracy'].log(logs['val_accuracy'])
            self.neptune_run['epoch/loss'].log(logs['loss'])    
            self.neptune_run['epoch/val_loss'].log(logs['val_loss'])
        else:
            train_acc_list = []
            val_acc_list = []
            train_loss_list = []
            val_loss_list = []

            for req in self.prop_args['reqs']:
                self.neptune_run[f'epoch/{req.value}/accuracy'].log(logs[f'{req.value}_accuracy'])
                self.neptune_run[f'epoch/{req.value}/val_accuracy'].log(logs[f'val_{req.value}_accuracy'])
                self.neptune_run[f'epoch/{req.value}/loss'].log(logs[f'{req.value}_loss'])
                self.neptune_run[f'epoch/{req.value}/val_loss'].log(logs[f'val_{req.value}_loss'])
                
                train_acc_list.append(logs[f'{req.value}_accuracy'])
                val_acc_list.append(logs[f'val_{req.value}_accuracy'])
                train_loss_list.append(logs[f'loss'])
                val_loss_list.append(logs[f'val_loss'])
            
            total_acc, total_val_acc = np.mean(train_acc_list), np.mean(val_acc_list)
            total_loss, total_val_loss = np.mean(train_loss_list), np.mean(val_loss_list)

            self.neptune_run['epoch/total_accuracy'].log(total_acc)
            self.neptune_run['epoch/total_val_accuracy'].log(total_val_acc)
            self.neptune_run[f'epoch/total_loss'].log(total_loss)
            self.neptune_run[f'epoch/total_val_loss'].log(total_val_loss)
    

    def __get_log_data_callback(self):
        return LambdaCallback(on_epoch_end = lambda epoch, logs: self.__log_data(logs))


    def __lr_scheduler(self, epoch):
            if epoch <= 10:
                new_lr = self.net_args['learning_rate']
            elif epoch <= 20:
                new_lr = self.net_args['learning_rate'] * 0.5
            elif epoch <= 40:
                new_lr = self.net_args['learning_rate'] * 0.5
            else:
                new_lr = self.net_args['learning_rate'] * np.exp(0.1 * ((epoch//self.net_args['n_epochs'])*self.net_args['n_epochs'] - epoch))

            if self.use_neptune:
                self.neptune_run['learning_rate'].log(new_lr)

            return new_lr


    def __get_lr_scheduler_callback(self):
        return LearningRateScheduler(self.__lr_scheduler)


    def __get_early_stopping_callback(self):
        return EarlyStopping(patience=self.net_args['early_stopping'], 
                            monitor='val_loss', 
                            restore_best_weights=True)


    def __get_model_checkpoint_callback(self):
        return ModelCheckpoint(self.checkpoint_path, 
                                monitor='val_loss', 
                                save_best_only=True, 
                                mode='min',
                                verbose=1)
        
    def get_callbacks_list(self):
        cust_optimizers_list = [Optimizer.ADAMAX_CUST.name, Optimizer.ADAGRAD_CUST.name,Optimizer.ADAM_CUST.name, Optimizer.SGD_CUST.name]

        callbacks_list = []
        if self.use_neptune:
            callbacks_list.append(self.__get_log_data_callback())
        
        if self.net_args['optimizer'].name in cust_optimizers_list:
            callbacks_list.append(self.__get_lr_scheduler_callback()) 

        if self.net_args['early_stopping'] is not None:
            callbacks_list.append(self.__get_early_stopping_callback()) 
        
        callbacks_list.append(self.__get_model_checkpoint_callback())

        return callbacks_list