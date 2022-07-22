import os
import shutil
import zipfile

import neptune.new as neptune

from src.m_utils.utils import print_method_log_sig

class NeptuneUtils:
    def __init__(self, config_interp):
        self.config_interp = config_interp

        self.orig_model_experiment_id = self.config_interp.prop_args['orig_model_experiment_id']
        self.is_training_model = self.config_interp.prop_args['train_model']

        self.neptune_run = None
        self.__start_neptune()


    def __start_neptune(self):
        print_method_log_sig(' starting neptune ')
        if self.config_interp.use_neptune:
            print('Starting Neptune')
            self.neptune_run = neptune.init(name=self.config_interp.exp_args['name'],
                                            description=self.config_interp.exp_args['description'],
                                            tags=self.config_interp.exp_args['tags'],
                                            source_files=self.config_interp.exp_args['src_files'])    
            print('----')
        else:
            print('Not using Neptune to record Experiment Metadata')


    def __check_prev_run_fields_benchmark(self, prev_run):
        prev_run_ds = prev_run['properties/benchmark_dataset'].fetch()
        prev_run_tasks = prev_run['properties/benchmark_tasks'].fetch()

        cur_run_ds = str(self.config_interp.prop_args['benchmarking']['benchmark_dataset'].name)
        cur_run_tasks = str([x.name for x in self.config_interp.prop_args['benchmarking']['tasks']])

        print(f' ...Prev Exp | Dataset: {prev_run_ds}')
        print(f' ...Prev Exp | Tasks: {prev_run_tasks}')
        print(f'')
        print(f' ...Current Exp | Dataset: {cur_run_ds}')
        print(f' ...Current Exp | Tasks: {cur_run_tasks}')

        if prev_run_ds != cur_run_ds:
            raise Exception('Previous experiment Dataset field does not match current experiment Dataset field!')
        if prev_run_tasks != cur_run_tasks:
            raise Exception('Previous experiment Tasks field does not match current experiment Tasks field!')


    def __check_prev_run_fields_icao(self, prev_run):
        icao_data = self.config_interp.prop_args['icao_data']
        icao_gt, reqs, aligned = icao_data['icao_gt'], icao_data['reqs'], icao_data['aligned']
        
        prev_run_req = prev_run['properties/icao_reqs'].fetch()
        prev_run_aligned = float(prev_run['properties/aligned'].fetch())
        prev_run_ds = prev_run['properties/gt_names'].fetch()

        print(f' ...Prev Exp | Req: {prev_run_req}')
        print(f' ...Prev Exp | Aligned: {prev_run_aligned}')
        print(f' ...Prev Exp | DS: {prev_run_ds}')
        
        if not self.config_interp.is_mtl_model:
            cur_run_req = str([reqs[0].value])
        else:
            cur_run_req = str([req.value for req in reqs])
        cur_run_aligned = float(int(aligned))
        gt_names_formatted = {
            'train_validation': [x.value.lower() for x in icao_gt['gt_names']['train_validation']],
            'test': [x.value.lower() for x in icao_gt['gt_names']['test']],
            'train_validation_test': [x.value.lower() for x in icao_gt['gt_names']['train_validation_test']]
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


    def __check_prev_run_fields(self):
        try:
            print('-----')
            print(' ..Checking previous experiment metadata')

            prev_run = None
            prev_run = neptune.init(run=self.orig_model_experiment_id)

            if self.config_interp.use_icao_gt:
                self.__check_prev_run_fields_icao(prev_run)
            elif self.config_interp.use_benchmark_data:
                self.__check_prev_run_fields_benchmark(prev_run)
            elif self.config_interp.use_icao_dl:
                raise NotImplemented()

            print(' ..All checked!')
            print('-----')
        
        except Exception as e:
            raise e
        finally:
            if prev_run is not None:
                prev_run.stop()
    
    
    def __download_model(self, trained_model_dir_path):
        try:
            print(f'Trained model dir path: {trained_model_dir_path}')
            prev_run = None
            prev_run = neptune.init(run=self.orig_model_experiment_id)

            print(f'..Downloading model from Neptune')
            print(f'..Experiment ID: {self.orig_model_experiment_id}')
            print(f'..Destination Folder: {trained_model_dir_path}')

            os.makedirs(trained_model_dir_path, exist_ok=True)
            destination_folder = trained_model_dir_path

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
    

    def check_model_existence(self, trained_model_dir_path):
        print('----')   
        print('Checking model existence locally...')
        if self.is_training_model:
            print('Training a new model! Not checking model existence')
        else:
            if os.path.exists(trained_model_dir_path):
                print('Model already exists locally. Not downloading!')
                print(f'Trained model dir path: {trained_model_dir_path}')
                self.__check_prev_run_fields()
            else:
                self.__check_prev_run_fields()
                self.__download_model(trained_model_dir_path)
        print('----')


    # download accuracy and loss series data from neptune
    # (previous experiment) and log them to current experiment
    def get_acc_and_loss_data(self):
        try:
            print(f' ..Experiment ID: {self.orig_model_experiment_id}')
            print(f' ..Downloading data from previous experiment')
            prev_run = neptune.init(run=self.orig_model_experiment_id)
            
            if not self.config_interp.is_mtl_model:
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

                for req in self.config_interp.prop_args['icao_data']['reqs']:
                    print(f' ..Requisite: {req}')
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
    

    # download neural architecture search data from neptune
    # (previous experiment) and log them to current experiment
    def get_nas_data(self, n_trials):
        try:
            print(f' ..Experiment ID: {self.orig_model_experiment_id}')
            print(f' ..Downloading data from previous experiment')
            prev_run = neptune.init(run=self.orig_model_experiment_id)
            
            print('  ...fetching data from previous trials')
            for i in range(1,n_trials+1):
                for n in range(4):
                    self.neptune_run[f'nas/trial/{i}/config/n_denses_{n}'] = prev_run[f'nas/trial/{i}/config/n_denses_{n}'].fetch()
                self.neptune_run[f'nas/trial/{i}/result/final_ACC'] = prev_run[f'nas/trial/{i}/result/final_ACC'].fetch()
                self.neptune_run[f'nas/trial/{i}/result/final_EER_mean'] = prev_run[f'nas/trial/{i}/result/final_EER_mean'].fetch()
            
            print('  ...fetching data from best trial')
            self.neptune_run[f'nas/best_trial/final_EER_mean'] = prev_run[f'nas/best_trial/final_EER_mean'].fetch()
            self.neptune_run[f'nas/best_trial/final_ACC'] = prev_run[f'nas/best_trial/final_ACC'].fetch()
            self.neptune_run[f'nas/best_trial/num'] = prev_run[f'nas/best_trial/num'].fetch()
            for n in range(4):
                self.neptune_run[f'nas/best_trial/config/n_denses_{n}'] = prev_run[f'nas/best_trial/config/n_denses_{n}'].fetch()

            print(f' ..Data uploaded to current experiment')
        except Exception as e:
            print('Error in get_nas_data()')
            raise e
        finally:
            prev_run.stop()