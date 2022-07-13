import os
import zipfile

from pathlib import Path

from IPython.display import display

# disable tensorflow log level infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from src.base.experiment.model_creator import ModelCreator
from src.base.experiment.model_train_visualizer import ModelTrainVisualizer
from src.base.experiment.train_callbacks import *


## restrict memory growth -------------------
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try:
    gpu_0 = physical_devices[0]
    tf.config.experimental.set_memory_growth(gpu_0, True) 
    #tf.config.experimental.set_virtual_device_configuration(gpu_0, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6500)])
    print(' ==> Restrict GPU memory growth: True')
except: 
    raise Exception("Invalid device or cannot modify virtual devices once initialized.")
## restrict memory growth ------------------- 


class ModelTrainer:
    def __init__(self, config_interp, neptune_utils):
        self.config_interp = config_interp
        self.neptune_utils = neptune_utils

        self.baseModel = None  # instance of base model keras/tensorflow
        
        self.is_training_model = self.config_interp.prop_args['train_model']
        
        self.orig_model_experiment_id = self.config_interp.prop_args['orig_model_experiment_id']

        self.running_nas = False
        
        self.CHECKPOINT_PATH = os.path.join('training_ckpt', 'best_model.hdf5')
        self.TRAINED_MODEL_DIR_PATH = None
        
        self.__set_model_path()
        self.__check_model_existence()
        self.__clear_checkpoints()
        self.__check_gpu_availability()

        self.model_creator = ModelCreator(self.config_interp)
        self.cb_handler = CallbacksHandler(self.config_interp, self.neptune_utils, self.CHECKPOINT_PATH)
        self.model_train_viz = ModelTrainVisualizer(self.config_interp)
        

    def set_running_nas_mode(self, running_nas):
        self.running_nas = running_nas

    
    def __set_model_path(self):
        model_path = None
        if self.orig_model_experiment_id != '':
            ds = self.config_interp.prop_args['icao_data']['icao_gt']['gt_names']['train_validation_test'][0].value
            aligned = 'aligned' if self.config_interp.prop_args['icao_data']['aligned'] else 'not_aligned'
            model_type = 'single_task' if not self.config_interp.is_mtl_model else 'multi_task'
            req = self.config_interp.prop_args['icao_data']['reqs'][0].value if not self.config_interp.is_mtl_model else 'multi_reqs'
            model_path = os.path.join('prev_trained_models', f'{model_type}', f'{ds}_{aligned}', f'{req}', f'{self.orig_model_experiment_id}')
        else:
            if not self.is_training_model:
                raise Exception('Insert orig_model_experiment_id in field of kwargs or train a new model!')
            else:
                model_path = os.path.join('trained_model')
        
        self.TRAINED_MODEL_DIR_PATH = model_path
       
    
    def __check_model_existence(self):
        self.neptune_utils.check_model_existence(self.TRAINED_MODEL_DIR_PATH)

    
    def __check_gpu_availability(self):
        print('------------------------------')
        print('Checking GPU availability')
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print(' ..GPU is available!')
        else:
            print(' ..GPU is NOT available!')
        print('------------------------------')
    
    
    def __clear_checkpoints(self):
        ckpts_dir_path = Path(self.CHECKPOINT_PATH.split(os.path.sep)[0])
        if(not os.path.exists(ckpts_dir_path)):
            ckpts_dir_path.mkdir(parents=True)
        
        if os.path.exists(self.CHECKPOINT_PATH):
            os.remove(self.CHECKPOINT_PATH)
    
    
    def create_model(self, train_gen=None, config=None, running_nas=False):
        if not self.is_training_model:
            print('Not creating a model: not training a model! ')
            return
        
        print('Creating model...')
        self.baseModel, self.model = self.model_creator.create_model(train_gen, config)

        if self.config_interp.use_neptune and not running_nas:
            self.model.summary(print_fn=lambda x: self.neptune_utils.neptune_run['summary/train/model_summary'].log(x))
        
        print('Model created')

        
    def model_summary(self, fine_tuned=False, print_fn=print):
        if self.is_training_model:
            self.model.summary(print_fn=print_fn)
        
            if self.config_interp.use_neptune:
                if not fine_tuned:
                    self.model.summary(print_fn=lambda x: self.neptune_utils.neptune_run['summary/train/model_summary'].log(x))
                else:
                    self.model.summary(print_fn=lambda x: self.neptune_utils.neptune_run['summary/train/fine_tune_model_summary'].log(x))
        else:
            print('Not training a model!')
           
    
    def visualize_model(self, outfile_path=None, verbose=True):
        if self.is_training_model:
            if verbose:
                display(plot_model(self.model, show_shapes=True, to_file=outfile_path))
            else:
                plot_model(self.model, show_shapes=True, to_file=outfile_path)
            
            if self.config_interp.use_neptune:
                self.neptune_utils.neptune_run['viz/model_architecture'].upload(outfile_path)
        else:
            print('Not training a model! No model to exhibit!')

            
    def __setup_fine_tuning(self, fine_tuned):
        if fine_tuned:
            print(' .. Fine tuning base model...')
            non_traininable_layers = self.baseModel.layers[:-2]
            
            print(f' .. Base model non trainable layers: {[l.name for l in non_traininable_layers]}')
            for layer in self.model.layers:
                if layer in non_traininable_layers:
                    layer.trainable = False
                else:
                    layer.trainable = True
        else:
            print(' .. Not fine tuning base model...')
            base_model_layers = [l.name for l in self.baseModel.layers]
            for m_l in self.model.layers:
                 if m_l.name in base_model_layers:
                    m_l.trainable = False
        
        def p_func(line):
            if 'params' in line.lower():
                print(f'  .. {line}')
        
        self.model_summary(fine_tuned, print_fn=p_func)
            

    def train_model(self, train_gen, validation_gen, fine_tuned, n_epochs, running_nas):
        if self.is_training_model:
            print(f'Training {self.config_interp.base_model.name} network')
            
            self.__setup_fine_tuning(fine_tuned)       

            callbacks_list = self.cb_handler.get_callbacks_list(running_nas)
            
            if n_epochs is None:
                epchs = self.config_interp.net_args['n_epochs']
            else:
                epchs = n_epochs
                if self.config_interp.use_neptune:
                    self.neptune_utils.neptune_run['parameters/n_epochs_fine_tuning'] = epchs
            
            vrb = 0 if running_nas else 1

            self.H = self.model.fit(
                    train_gen,
                    steps_per_epoch=train_gen.n // self.config_interp.net_args['batch_size'],
                    validation_data=validation_gen,
                    validation_steps=validation_gen.n // self.config_interp.net_args['batch_size'],
                    epochs=epchs,
                    verbose=vrb,
                    callbacks=callbacks_list)
        
        elif not self.is_training_model and self.config_interp.use_neptune:
            print(f'Not training a model. Downloading data from Neptune')
            self.neptune_utils.get_acc_and_loss_data()
        else:
            print(f'Not training a model and not using Neptune!')
       
    
    def draw_training_history(self):
        if self.is_training_model:
            f = self.model_train_viz.visualize_history(self.H)

            if self.config_interp.use_neptune:
                self.neptune_utils.neptune_run['viz/train/training_curves'].upload(f)
        
        elif not self.is_training_model and self.config_interp.use_neptune:
            print('Not training a model. Downloading plot from Neptune')
            self.neptune_utils.get_training_curves()
        
        else:
            print('Not training a model and not using Neptune!')
    

    def load_checkpoint(self, chkp_name):
        self.__create_model()
        self.model.load_weights(chkp_name)
    
    
    def load_best_model(self):
        print('..Loading best model')
        
        if self.is_training_model:
            if os.path.isfile(self.CHECKPOINT_PATH):
                self.model.load_weights(self.CHECKPOINT_PATH)
                print('..Checkpoint weights loaded')
            else:
                print('Checkpoint not found')
        else:
            self.model = load_model(self.TRAINED_MODEL_DIR_PATH)
            print('..Model loaded')
            print(f'...Model path: {self.TRAINED_MODEL_DIR_PATH}')
            
    
    def save_trained_model(self):
        if self.config_interp.prop_args['save_trained_model']:
            def zipdir(path):
                outfile_path = 'trained_model.zip'
                with zipfile.ZipFile(outfile_path, 'w', zipfile.ZIP_DEFLATED) as ziph:
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            filename = os.path.join(root, file)
                            arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                            if self.orig_model_experiment_id != "" and self.orig_model_experiment_id in arcname:
                                arcname = arcname.replace(self.orig_model_experiment_id, 'trained_model')
                            ziph.write(filename, arcname)
                return outfile_path
            
            print('Saving model')

            self.model.save(self.TRAINED_MODEL_DIR_PATH)
            print('..Model saved')
            print(f'...Model path: {self.TRAINED_MODEL_DIR_PATH}')

            if self.config_interp.use_neptune:
                print('Saving model to neptune')
                trained_model_zip_path = zipdir(self.TRAINED_MODEL_DIR_PATH)
                print(f' ..Uploading file {trained_model_zip_path}')
                self.neptune_utils.neptune_run['artifacts/trained_model'].upload(trained_model_zip_path)
                print('Model saved into Neptune')

            self.model.training = False

            print('Saving process finished')
        else:
            print('Not saving trained model!')
        