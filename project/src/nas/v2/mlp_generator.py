import os
import warnings
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

from src.nas.v2.mlp_search_space import MLPSearchSpace
from src.base.experiment.training.optimizers import Optimizer

class MLPGenerator(MLPSearchSpace):

    def __init__(self, config_interp):

        self.config_interp = config_interp

        self.mlp_optimizer  = self.config_interp.mlp_params['mlp_optimizer']
        self.mlp_lr         = self.config_interp.mlp_params['mlp_learning_rate']
        self.mlp_decay      = self.config_interp.mlp_params['mlp_decay']
        self.mlp_momentum   = self.config_interp.mlp_params['mlp_momentum']
        self.mlp_dropout    = self.config_interp.mlp_params['mlp_dropout']
        self.mlp_loss_func  = self.config_interp.mlp_params['mlp_loss_function']
        self.mlp_one_shot   = self.config_interp.mlp_params['mlp_one_shot']
        self.metrics        = ['accuracy']

        n_tasks = len(self.config_interp.prop_args['benchmarking']['dataset'].value['tasks'])

        super().__init__(n_tasks)


        if self.mlp_one_shot:
            self.weights_file = 'LOGS/shared_weights.pkl'
            self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})
            if not os.path.exists(self.weights_file):
                print("Initializing shared weights dictionary...")
                self.shared_weights.to_pickle(self.weights_file)


    def create_model(self, sequence, mlp_input_shape):
        layer_configs = self.decode_sequence(sequence)
        model = Sequential()
        if len(mlp_input_shape) > 1:
            model.add(Flatten(name='flatten', input_shape=mlp_input_shape))
            for i, layer_conf in enumerate(layer_configs):
                if layer_conf is 'dropout':
                    model.add(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))
        else:
            for i, layer_conf in enumerate(layer_configs):
                if i == 0:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1], input_shape=mlp_input_shape))
                elif layer_conf is 'dropout':
                    model.add(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))
        return model


    def compile_model(self, model):
        if self.mlp_optimizer.name == Optimizer.SGD.name:
            optim = optimizers.SGD(lr=self.mlp_lr, decay=self.mlp_decay, momentum=self.mlp_momentum)
        elif self.mlp_optimizer == Optimizer.SGD_NESTEROV.name:
            optim = optimizers.SGD(lr=self.mlp_lr, decay=self.mlp_decay, momentum=self.mlp_momentum, nesterov=True)
        else:
            optim = getattr(optimizers, self.mlp_optimizer.value)(lr=self.mlp_lr, decay=self.mlp_decay)

        print('Used optimizer: ', optim)

        model.compile(loss=self.mlp_loss_func, optimizer=optim, metrics=self.metrics)
        
        return model


    def update_weights(self, model):
        layer_configs = ['input']
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                if len(search_index) == 0:
                    self.shared_weights = self.shared_weights.append({'bigram_id': config_ids[j],
                                                                      'weights': layer.get_weights()},
                                                                     ignore_index=True)
                else:
                    self.shared_weights.at[search_index[0], 'weights'] = layer.get_weights()
                j += 1
        self.shared_weights.to_pickle(self.weights_file)


    def set_model_weights(self, model):
        layer_configs = ['input']
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                if len(search_index) > 0:
                    print("Transferring weights for layer:", config_ids[j])
                    layer.set_weights(self.shared_weights['weights'].values[search_index[0]])
                j += 1


    def train_model(self, model, train_gen, validation_gen, nb_epochs, callbacks=None):
        if self.mlp_one_shot:
            self.set_model_weights(model)
            history = model.fit(train_gen,
                                validation_data=validation_gen,
                                epochs=nb_epochs,
                                callbacks=callbacks,
                                verbose=0)
            self.update_weights(model)
        else:
            history = model.fit(train_gen,
                                validation_data=validation_gen,
                                epochs=nb_epochs,
                                callbacks=callbacks,
                                verbose=0)
        return history
