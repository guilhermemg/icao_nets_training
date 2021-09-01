import os
import shutil
import zipfile

import neptune.new as neptune


class NeptuneUtils:
    def __init__(self, prop_args, orig_model_experiment_id, is_mtl_model, trained_model_dir_path, is_training_model, neptune_run):
        self.prop_args = prop_args
        self.orig_model_experiment_id = orig_model_experiment_id
        self.is_mtl_model = is_mtl_model
        self.trained_model_dir_path = trained_model_dir_path
        self.is_training_model = is_training_model
        self.neptune_run = neptune_run


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
            print(f'Trained model dir path: {self.trained_model_dir_path}')
            prev_run = None
            prev_run = neptune.init(run=self.orig_model_experiment_id)

            print(f'..Downloading model from Neptune')
            print(f'..Experiment ID: {self.orig_model_experiment_id}')
            print(f'..Destination Folder: {self.trained_model_dir_path}')

            os.mkdir(self.trained_model_dir_path)
            destination_folder = self.trained_model_dir_path

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
    

    def check_model_existence(self):
        print('----')   
        print('Checking model existence locally...')
        if self.is_training_model:
            print('Training a new model! Not checking model existence')
        else:
            if os.path.exists(self.trained_model_dir_path):
                print('Model already exists locally. Not downloading!')
                print(f'Trained model dir path: {self.trained_model_dir_path}')
                self.__check_prev_run_fields()
            else:
                self.__check_prev_run_fields()
                self.__download_model()
        print('----')


    # download accuracy and loss series data from neptune
    # (previous experiment) and log them to current experiment
    def get_acc_and_loss_data(self):
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
                total_acc_series = prev_run[f'epoch/total_accuracy'].fetch_values()['value']
                total_val_acc_series = prev_run[f'epoch/total_val_accuracy'].fetch_values()['value']
                total_loss_series = prev_run[f'epoch/total_loss'].fetch_values()['value']

                for(acc,val_acc) in zip(total_acc_series,total_val_acc_series):
                    self.neptune_run[f'epoch/total_accuracy'].log(acc)
                    self.neptune_run[f'epoch/total_val_accuracy'].log(val_acc)
                    
                for ls in total_loss_series:
                    self.neptune_run[f'epoch/total_loss'].log(ls)

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
    

    # download training curves plot from Neptune previous experiment
    # and upload it to the current one
    def get_training_curves(self):
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