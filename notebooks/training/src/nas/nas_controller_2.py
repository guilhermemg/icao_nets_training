
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal, RandomUniform, GlorotNormal

from nas.gen_nas_controller import GenNASController
from utils.constants import SEED

class NASController_2(GenNASController):
    def __init__(self, nas_params, neptune_run, use_neptune):
        super().__init__(nas_params, neptune_run, use_neptune)       
        
        self.baseline = None
        self.lstm_cell_units = 32
        self.baseline_decay = 0.999
        self.opt = Adam(lr=0.00035, decay=1e-3, amsgrad=True)
        self.controller_batch_size = self.nas_params['controller_batch_size']
        self.controller_epochs = self.nas_params['controller_epochs']

        self.input_x = np.array([[[SEED,SEED,SEED,SEED]]])

        self.reward = 0

        self.controller_rnn = None
        
        self.__generate_controller_rnn()


    def __generate_controller_rnn(self):
        controller_input = Input(shape=(1,4,))        

        cell = LSTMCell(
            self.lstm_cell_units,
            kernel_initializer=self.__get_weight_initializer(initializer="lstm"),
            recurrent_initializer=self.__get_weight_initializer(initializer="lstm"))
        
        x = RNN(cell, return_state=True)(controller_input)
        y = Dense(4)(x[0])
        y = Activation(activation="softmax")(y)
        
        self.controller_rnn = Model(inputs=controller_input, outputs=y)

    
    def __compile_controller_rnn(self):
        def __controller_loss(y_true, y_pred):
            if self.baseline is None:
                self.baseline = 0
            else:
                self.baseline -= (1 - self.baseline_decay) * (self.baseline - self.reward)
            return y_pred * (self.reward - self.baseline)

        self.controller_rnn.compile(loss=__controller_loss, optimizer=self.opt)


    def train_controller_rnn(self, targets):
        print(f' .. training controller rnn ..')
        print(f'  .. targets: {targets}')
        self.__compile_controller_rnn()
        self.controller_rnn.fit(
            self.input_x,
            targets,
            epochs=self.controller_epochs,
            batch_size=self.controller_batch_size,
            verbose=1)


    def __softmax_predict(self):
        self.__compile_controller_rnn()
        return self.controller_rnn.predict(self.input_x)


    def __convert_pred_to_ydict(self, controller_pred):
        vals = controller_pred[0]
        vals = np.interp(vals, (vals.min(), vals.max()), (1, self.MAX_BLOCKS_PER_BRANCH))
        vals = vals.astype('int')
        config = {f'n_denses_0': vals[0], f'n_denses_1': vals[1], f'n_denses_2': vals[2], f'n_denses_3': vals[3]}
        return config


    def select_config(self):
        print(' selecting new config...')
        controller_pred = self.__softmax_predict()
        print(f' controller_pred: {controller_pred}')

        config = self.__convert_pred_to_ydict(controller_pred)     
        
        self.cur_trial.set_config(config)

        return controller_pred, config
    

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


    def __get_weight_initializer(self, initializer=None, seed=None):
        if initializer is None:
            return HeNormal()
        elif initializer == "lstm":
            return RandomUniform(minval=-0.1, maxval=0.1)
        else:
            return GlorotNormal()




# class ControllerRNNController(object):
#     def __init__(self,
#                  controller_network_name,
#                  num_nodes,
#                  num_opers,
#                  input_x,
#                  reward=0,
#                  temperature=5.0,
#                  tanh_constant=2.5,
#                  model_file=None,
#                  lstm_cell_units=32,
#                  baseline_decay=0.999,
#                  opt=Adam(learning_rate=0.00035, decay=1e-3, amsgrad=True)):

#         self.controller_network_name = controller_network_name
#         self.num_nodes = num_nodes
#         self.num_opers = num_opers
#         self.reward = reward
#         self.input_x = input_x
#         self.temperature = temperature
#         self.tanh_constant = tanh_constant
#         self.lstm_cell_units = lstm_cell_units
#         self.opt = opt
#         self.model_file = model_file

#         self.controller_rnn = self.generate_controller_rnn()
#         self.baseline = None
#         self.baseline_decay = baseline_decay

#         #self.graph = tf.get_default_graph()

#     def lstm_reshape(self,
#                      inputs,
#                      name_prefix,
#                      index,
#                      reshaped_inputs=None,
#                      initial=False):
#         name_prefix = "{0}_{1}_{2}".format(self.controller_network_name,
#                                            name_prefix, index)
#         cell = LSTMCell(
#             self.lstm_cell_units,
#             kernel_initializer=get_weight_initializer(initializer="lstm"),
#             recurrent_initializer=get_weight_initializer(initializer="lstm"))
#         if initial:
#             x = RNN(
#                 cell,
#                 return_state=True,
#                 name="{0}_{1}".format(name_prefix, "lstm"))(inputs)
#         else:
#             x = RNN(
#                 cell,
#                 return_state=True,
#                 name="{0}_{1}".format(name_prefix, "lstm"))(
#                     reshaped_inputs, initial_state=inputs[1:])
#         rx = Reshape(
#             (-1, self.lstm_cell_units),
#             name="{0}_{1}".format(name_prefix, "reshape"))(x[0])
#         return x, rx

#     def dense_softmax(self, inputs, num_classes, name_prefix, index):
#         name_prefix = "{0}_{1}_{2}".format(self.controller_network_name,
#                                            name_prefix, index)
#         y = Dense(
#             num_classes, name="{0}_{1}".format(name_prefix, "dense"))(inputs)
#         y = Activation(
#             activation="softmax",
#             name="{0}_{1}".format(name_prefix, "softmax"))(y)
#         return y

#     def generate_controller_rnn(self):
#         outputs = []
#         controller_input = Input(shape=(1, 1,), name="{0}_{1}".format(self.controller_network_name, "input"))

#         for i in range(2, self.num_nodes):
#             for o in ["inputL", "inputR", "operL", "operR"]:
#                 if i == 2 and o == "inputL":
#                     _x, _rx, _initial = controller_input, None, True
#                 else:
#                     _x, _rx, _initial = x, rx, False

#                 if o in ["inputL", "inputR"]:
#                     _num_classes = i
#                 else:
#                     _num_classes = self.num_opers

#                 x, rx = self.lstm_reshape(
#                     inputs=_x,
#                     name_prefix=o,
#                     index=i,
#                     reshaped_inputs=_rx,
#                     initial=_initial)
#                 y = self.dense_softmax(
#                     inputs=x[0],
#                     num_classes=_num_classes,
#                     name_prefix=o,
#                     index=i)
#                 outputs.append(y)

#         controller_rnn = Model(inputs=controller_input, outputs=outputs)

#         if self.model_file is not None and os.path.exists(self.model_file):
#             controller_rnn.load_weights(self.model_file)
#         return controller_rnn

#     def compile_controller_rnn(self):
#         def _controller_loss(y_true, y_pred):
#             if self.baseline is None:
#                 self.baseline = 0
#             else:
#                 self.baseline -= (1 - self.baseline_decay) * (self.baseline - self.reward)
#             return y_pred * (self.reward - self.baseline)

#         def _define_loss(controller_loss):
#             outputs_loss = {}
#             for i in range(2, self.num_nodes):
#                 outputs_loss["{0}_{1}_{2}_{3}".format(self.controller_network_name, "inputL", i, "softmax")] = controller_loss
#                 outputs_loss["{0}_{1}_{2}_{3}".format(self.controller_network_name, "inputR", i, "softmax")] = controller_loss
#                 outputs_loss["{0}_{1}_{2}_{3}".format(self.controller_network_name, "operL", i, "softmax")] = controller_loss
#                 outputs_loss["{0}_{1}_{2}_{3}".format(self.controller_network_name, "operR", i, "softmax")] = controller_loss
#             return outputs_loss

#         self.controller_rnn.compile(loss=_define_loss(_controller_loss), optimizer=self.opt)

#     def save_model(self):
#         self.controller_rnn.save_weights(self.model_file)

#     def train_controller_rnn(self,
#                              targets,
#                              batch_size=1,
#                              epochs=50,
#                              callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')]):
#         #with self.graph.as_default():
#         self.compile_controller_rnn()
#         self.controller_rnn.fit(
#             self.input_x,
#             targets,
#             epochs=epochs,
#             batch_size=batch_size,
#             verbose=0)

#     def softmax_predict(self):
#         #with self.graph.as_default():
#         self.compile_controller_rnn()
#         return self.controller_rnn.predict(self.input_x)

#     def random_sample_softmax(self, controller_pred):
#         sample_softmax = []
#         for cp in controller_pred:
#             cp /= self.temperature
#             cp = self.tanh_constant * np.tanh(cp)
#             cp = np.exp(cp) / np.sum(np.exp(cp))
#             cp = np.array([np.random.multinomial(1, cp[0])])
#             sample_softmax.append(cp)
#         return sample_softmax

#     def convert_pred_to_cell(self, controller_pred):
#         cell_pred = {}
#         for p in range(2, self.num_nodes):
#             pos = list(range((p - 2) * 4, ((p - 2) * 4) + 4))
#             cell_pred[p] = {
#                 "L": {
#                     "input_layer": np.argmax(controller_pred[pos[0]]),
#                     "oper_id": np.argmax(controller_pred[pos[2]])
#                 },
#                 "R": {
#                     "input_layer": np.argmax(controller_pred[pos[1]]),
#                     "oper_id": np.argmax(controller_pred[pos[3]])
#                 }
#             }
#         return cell_pred

#     def convert_pred_to_ydict(self, controller_pred):
#         ydict = {}
#         name_prefix = self.controller_network_name
#         for i in range(2, self.num_nodes):
#             pos = list(range((i - 2) * 4, ((i - 2) * 4) + 4))
#             ydict["{0}_{1}_{2}_{3}".format(name_prefix, "inputL", i, "softmax")] = controller_pred[pos[0]]
#             ydict["{0}_{1}_{2}_{3}".format(name_prefix, "inputR", i, "softmax")] = controller_pred[pos[1]]
#             ydict["{0}_{1}_{2}_{3}".format(name_prefix, "operL", i, "softmax")] = controller_pred[pos[2]]
#             ydict["{0}_{1}_{2}_{3}".format(name_prefix, "operR", i, "softmax")] = controller_pred[pos[3]]
#         return ydict
