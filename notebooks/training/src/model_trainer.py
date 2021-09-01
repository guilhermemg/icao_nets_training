import os
import shutil
import zipfile

import neptune.new as neptune

from pathlib import Path

from IPython.display import display

# disable tensorflow log level infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from model_creator import ModelCreator
from model_train_visualizer import ModelTrainVisualizer
from train_callbacks import *


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
    def __init__(self, net_args, prop_args, base_model, is_mtl_model, mtl_approach, neptune_run):
        self.net_args = net_args
        self.prop_args = prop_args
        self.is_mtl_model = is_mtl_model
        self.mtl_approach = mtl_approach
        self.neptune_run = neptune_run
        self.use_neptune = True if neptune_run is not None else False
        self.base_model = base_model  # object of type enum BaseModel 
        
        self.baseModel = None  # instance of base model keras/tensorflow
        
        self.is_training_model = self.prop_args['train_model']
        
        self.orig_model_experiment_id = self.prop_args['orig_model_experiment_id']
        
        self.CHECKPOINT_PATH = os.path.join('training_ckpt', 'best_model.hdf5')
        self.TRAINED_MODEL_DIR_PATH = None
        
        self.__set_model_path()
        self.__check_model_existence()
        self.__clear_checkpoints()
        self.__check_gpu_availability()

        self.model_creator = ModelCreator(self.net_args, self.prop_args, self.base_model, self.mtl_approach, self.is_mtl_model)
        self.cb_handler = CallbacksHandler(self.net_args, self.prop_args, self.use_neptune, self.neptune_run, self.CHECKPOINT_PATH, self.is_mtl_model)
        self.model_train_viz = ModelTrainVisualizer(self.prop_args, self.base_model, self.is_mtl_model)
        
    
    def __set_model_path(self):
        model_path = None
        if self.orig_model_experiment_id != '':
            ds = self.prop_args['gt_names']['train_validation_test'][0].value
            aligned = 'aligned' if self.prop_args['aligned'] else 'not_aligned'
            model_type = 'single_task' if not self.is_mtl_model else 'multi_task'
            req = self.prop_args['reqs'][0].value if not self.is_mtl_model else 'multi_reqs'
            model_path = os.path.join('prev_trained_models', f'{model_type}', f'{ds}_{aligned}', f'{req}', f'{self.orig_model_experiment_id}')
        else:
            if not self.is_training_model:
                raise Exception('Insert orig_model_experiment_id in field of kwargs or train a new model!')
            else:
                model_path = os.path.join('trained_model')
        
        self.TRAINED_MODEL_DIR_PATH = model_path
    

    def __check_prev_run_fields(self):
        try:
            print('-----')
            print(' ..Checking previous experiment metadata')
            
            prev_run = None
            prev_run = neptune.init(run=self.orig_model_experiment_id)
            
            prev_run_req = prev_run['properties/icao_reqs'].fetch()
            prev_run_aligned = float(prev_run['properties/aligned'].fetch())
            prev_run_ds = prev_run['properties/gt_names'].fetch()

            print(f' ...Prev Exp | Req: {prev_run_req}')
            print(f' ...Prev Exp | Aligned: {prev_run_aligned}')
            print(f' ...Prev Exp | DS: {prev_run_ds}')
            
            if not self.is_mtl_model:
                cur_run_req = str([self.prop_args['reqs'][0].value])
            else:
                cur_run_req = str([req.value for req in self.prop_args['reqs']])
            cur_run_aligned = float(int(self.prop_args['aligned']))
            gt_names_formatted = {
                'train_validation': [x.value.lower() for x in self.prop_args['gt_names']['train_validation']],
                'test': [x.value.lower() for x in self.prop_args['gt_names']['test']],
                'train_validation_test': [x.value.lower() for x in self.prop_args['gt_names']['train_validation_test']]
            }
            cur_run_ds = str({'gt_names': str(gt_names_formatted)})

            print(f' ...Current Exp | Req: {cur_run_req}')
            print(f' ...Current Exp | Aligned: {cur_run_aligned}')
            print(f' ...Current Exp | DS: {cur_run_ds}')

            if prev_run_req != cur_run_req:
                raise Exception('Previous experiment Requisite field does not match current experiment Requisite field!')
            if prev_run_aligned != cur_run_aligned:
                raise Exception('Previous experiment Aligned field does not match current experiment Aligned field!')
            if prev_run_req != cur_run_req:
                raise Exception('Previous experiment DS fields does not match current experiment DS field!')

            print(' ..All checked!')
            print('-----')
        
        except Exception as e:
            raise e
        finally:
            if prev_run is not None:
                prev_run.stop()
    
    
    def __download_model(self):
        try:
            print(f'Trained model dir path: {self.TRAINED_MODEL_DIR_PATH}')
            prev_run = None
            prev_run = neptune.init(run=self.orig_model_experiment_id)

            print(f'..Downloading model from Neptune')
            print(f'..Experiment ID: {self.orig_model_experiment_id}')
            print(f'..Destination Folder: {self.TRAINED_MODEL_DIR_PATH}')

            os.mkdir(self.TRAINED_MODEL_DIR_PATH)
            destination_folder = self.TRAINED_MODEL_DIR_PATH

            prev_run['artifacts/trained_model'].download(destination_folder)
            print(' .. Download done!')

            with zipfile.ZipFile(os.path.join(destination_folder, 'trained_model.zip'), 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
            
            folder_name = [x for x in os.listdir(destination_folder) if '.zip' not in x][0]
            
            os.remove(os.path.join(destination_folder, 'trained_model.zip'))
            shutil.move(os.path.join(destination_folder, folder_name, 'variables'), destination_folder)
            shutil.move(os.path.join(destination_folder, folder_name, 'saved_model.pb'), destination_folder)
            shutil.rmtree(os.path.join(destination_folder, folder_name))

            print('.. Folders set')
            print('-----------------------------')
        except Exception as e:
            raise e
        finally:
            if prev_run is not None:
                prev_run.stop()
    
    
    def __check_model_existence(self):
        print('----')
        print('Checking model existence locally...')
        if self.is_training_model:
            print('Training a new model! Not checking model existence')
        else:
            if os.path.exists(self.TRAINED_MODEL_DIR_PATH):
                print('Model already exists locally. Not downloading!')
                print(f'Trained model dir path: {self.TRAINED_MODEL_DIR_PATH}')
                self.__check_prev_run_fields()
            else:
                self.__check_prev_run_fields()
                self.__download_model()
        print('----')
    
    
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
    
    
    def create_model(self, train_gen=None):
        print('Creating model...')
        
        self.baseModel, self.model = self.model_creator.create_model(train_gen)

        if self.use_neptune:
            self.model.summary(print_fn=lambda x: self.neptune_run['summary/train/model_summary'].log(x))
        
        print('Model created')

        
    def model_summary(self, fine_tuned=False, print_fn=print):
        if self.is_training_model:
            self.model.summary(print_fn=print_fn)
        
            if self.use_neptune:
                if not fine_tuned:
                    self.model.summary(print_fn=lambda x: self.neptune_run['summary/train/model_summary'].log(x))
                else:
                    self.model.summary(print_fn=lambda x: self.neptune_run['summary/train/fine_tune_model_summary'].log(x))
        else:
            print('Not training a model!')
           
    
    def vizualize_model(self, outfile_path=None):
        display(plot_model(self.model, show_shapes=True, to_file=outfile_path))
        if self.use_neptune:
            self.neptune_run['viz/model_architecture'].upload(outfile_path)

            
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
            

    def train_model(self, train_gen, validation_gen, fine_tuned, n_epochs):
        if self.is_training_model:
            print(f'Training {self.base_model.name} network')
            
            self.__setup_fine_tuning(fine_tuned)       

            callbacks_list = self.cb_handler.get_callbacks_list()
            
            if n_epochs is None:
                epchs = self.net_args['n_epochs']
            else:
                epchs = n_epochs
                if self.use_neptune:
                    self.neptune_run['parameters/n_epochs_fine_tuning'] = epchs

            self.H = self.model.fit(
                    train_gen,
                    steps_per_epoch=train_gen.n // self.net_args['batch_size'],
                    validation_data=validation_gen,
                    validation_steps=validation_gen.n // self.net_args['batch_size'],
                    epochs=epchs,
                    callbacks=callbacks_list)
        
        elif not self.is_training_model and self.use_neptune:
            print(f'Not training a model. Downloading data from Neptune')
            self.__get_acc_and_loss_data()
        
        else:
            print(f'Not training a model and not using Neptune!')

    
    # download accuracy and loss series data from neptune
    # (previous experiment) and log them to current experiment
    def __get_acc_and_loss_data(self):
        try:
            print(f' ..Experiment ID: {self.orig_model_experiment_id}')
            print(f' ..Downloading data from previous experiment')
            prev_run = neptune.init(run=self.orig_model_experiment_id)
            
            if not self.is_mtl_model:
                acc_series = prev_run['epoch/accuracy'].fetch_values()['value']
                val_acc_series = prev_run['epoch/val_accuracy'].fetch_values()['value']
                loss_series = prev_run['epoch/loss'].fetch_values()['value']
                val_loss_series = prev_run['epoch/val_loss'].fetch_values()['value']
                print(f' ..Download finished')

                print(f' ..Upload data to current experiment')
                for (acc,val_acc,loss,loss_val) in zip(acc_series,val_acc_series,loss_series,val_loss_series):
                    self.neptune_run['epoch/accuracy'].log(acc)
                    self.neptune_run['epoch/val_accuracy'].log(val_acc)
                    self.neptune_run['epoch/loss'].log(loss)    
                    self.neptune_run['epoch/val_loss'].log(loss_val)
            else:
                for req in self.prop_args['reqs']:
                    req = req.value
                    acc_series = prev_run[f'epoch/{req}/accuracy'].fetch_values()['value']
                    val_acc_series = prev_run[f'epoch/{req}/val_accuracy'].fetch_values()['value']
                    loss_series = prev_run[f'epoch/{req}/loss'].fetch_values()['value']
                    val_loss_series = prev_run[f'epoch/{req}/val_loss'].fetch_values()['value']
                    print(f' ..Download finished')

                    print(f' ..Upload data to current experiment')
                    for (acc,val_acc,loss,loss_val) in zip(acc_series,val_acc_series,loss_series,val_loss_series):
                        self.neptune_run[f'epoch/{req}/accuracy'].log(acc)
                        self.neptune_run[f'epoch/{req}/val_accuracy'].log(val_acc)
                        self.neptune_run[f'epoch/{req}/loss'].log(loss)    
                        self.neptune_run[f'epoch/{req}/val_loss'].log(loss_val)

            print(f' ..Upload finished')
        except Exception as e:
            print('Error in __get_acc_and_loss_data()')
            raise e
        finally:
            prev_run.stop()
        
    
    def draw_training_history(self):
        if self.is_training_model:
            f = self.model_train_viz.visualize_history(self.H)

            if self.use_neptune:
                self.neptune_run['viz/train/training_curves'].upload(f)
        
        elif not self.is_training_model and self.use_neptune:
            print('Not training a model. Downloading plot from Neptune')
            self.__get_training_curves()
        
        else:
            print('Not training a model and not using Neptune!')
    
    
    # download training curves plot from Neptune previous experiment
    # and upload it to the current one
    def __get_training_curves(self):
        try:
            print(f' ..Experiment ID: {self.orig_model_experiment_id}')
            print(f' ..Downloading plot from previous experiment')
            prev_run = neptune.init(run=self.orig_model_experiment_id)
            prev_run['viz/train/training_curves'].download()
            print(f' ..Download finished')

            print(f' ..Uploading plot')
            self.neptune_run['viz/train/training_curves'].upload('training_curves.png')
            print(f' ..Upload finished')
        except Exception as e:
            print('Error in __get_training_curves()')
            raise e
        finally:
            prev_run.stop()
    

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
        if self.prop_args['save_trained_model']:
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

            if self.use_neptune:
                print('Saving model to neptune')
                trained_model_zip_path = zipdir(self.TRAINED_MODEL_DIR_PATH)
                print(f' ..Uploading file {trained_model_zip_path}')
                self.neptune_run['artifacts/trained_model'].upload(trained_model_zip_path)
                print('Model saved into Neptune')

            self.model.training = False

            print('Saving process finished')
        else:
            print('Not saving trained model!')
        