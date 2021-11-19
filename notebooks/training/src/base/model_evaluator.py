import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from enum import Enum

import matplotlib.pyplot as plt

from neptune.new.types import File

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import tensorflow.keras.backend as K

from tensorflow.keras import preprocessing
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from gt_loaders.gen_gt import Eval

import utils.draw_utils as dr

from m_utils.constants import SEED, ICAO_REQ


class DataSource(Enum):
    VALIDATION = 'validation'
    TEST = 'test'  

    
class DataPredSelection(Enum):
    ANY = {'title': 'Any (TP,FP,TN,FN) images', 'abv': 'any_imgs'}
    ONLY_FP = {'title': 'Only False Positives images', 'abv': 'fp_only'}
    ONLY_FN = {'title': 'Only False Negatives images', 'abv': 'fn_only'}
    ONLY_TP = {'title': 'Only True Positives images', 'abv': 'tp_only'}
    ONLY_TN = {'title': 'Only True Negatives images', 'abv': 'tn_only'}



class FinalEvaluation:
    def __init__(self, evals_list):
        self.evals_list = evals_list
        self.final_EER_mean = None
        self.final_ACC = None
    
    def calculate_final_metrics(self):
        final_EER_mean = np.sum([r_ev.EER_mean for r_ev in self.evals_list])/len(self.evals_list)
        final_ACC = np.sum([r_ev.ACC for r_ev in self.evals_list])/len(self.evals_list)

        self.final_EER_mean = round(final_EER_mean * 100, 2)
        self.final_ACC = round(final_ACC * 100, 2)

        return {'final_EER_mean':self.final_EER_mean, 'final_ACC':self.final_ACC}

    def log_to_neptune(self, neptune_run, data_src):
        neptune_run[f'metrics/{data_src.value}/final_EER_mean'] = self.final_EER_mean
        neptune_run[f'metrics/{data_src.value}/final_ACC'] = self.final_ACC

    def __str__(self):
        return f'final_EER_mean: {self.final_EER_mean}% | final_ACC: {self.final_ACC}%'



class RequisiteEvaluation:
    def __init__(self, req):
        self.req = req
        self.TP = None
        self.FP = None
        self.TN = None
        self.FN = None
        self.EER_interp = None
        self.EER_mean = None
        self.best_th = None
        self.FAR = None
        self.FRR = None
        self.ACC = None
    
    def __str__(self) -> str:
        return f'Requisite: {self.req}\n' + \
                f' TP: {self.TP} | ' + \
                f' TN: {self.TN} | ' + \
                f' FP: {self.FP} | ' + \
                f' FN: {self.FN}\n' + \
                f' ACC: {self.ACC}\n' + \
                f' EER_interp: {self.EER_interp} | ' + \
                f' EER_mean: {self.EER_mean} | ' + \
                f' Best_thresh: {self.best_th} | ' + \
                f' FAR: {self.FAR} | ' + \
                f' FRR: {self.FRR}'


class ModelEvaluator:
    def __init__(self, net_args, prop_args, is_mtl_model, neptune_run):
        self.net_args = net_args
        self.prop_args = prop_args
        self.is_mtl_model = is_mtl_model
        self.neptune_run = neptune_run
        self.use_neptune = True if neptune_run is not None else False
        
        self.data_src = None
        
        self.metrics_var_base_path = None  # base path of metrics-variables in Neptune
        self.vis_var_base_path = None # base path of vizualizations-variables in Neptune
        
        self.y_test_true = None
        self.y_test_hat = None
        self.y_test_hat_discrete = None
        
        self.THRESH_START_VAL = 0.0
        self.THRESH_END_VAL = 1.02
        self.THRESH_STEP_SIZE = 1e-2
    
    
    def set_data_src(self, data_src):
        if data_src.value in [item.value for item in DataSource]:
            self.data_src = data_src
            self.metrics_var_base_path = f'metrics/{self.data_src.value.lower()}'
            self.vis_var_base_path = f'viz/{self.data_src.value.lower()}'
        else:
            raise Exception(f'Error! Invalid data source. Valid Options: {list(DataSource)}')
    
    
    def __calculate_far(self, y_true, y_pred):
        far = []
        n_non_comp = len([x for x in y_true if x == Eval.NON_COMPLIANT.value])
        th_range = np.arange(self.THRESH_START_VAL, self.THRESH_END_VAL, self.THRESH_STEP_SIZE) 
        for th in th_range:
            num = 0
            for tr_val,pred in zip(y_true,y_pred):
                if pred >= th and tr_val == Eval.NON_COMPLIANT.value:
                    num += 1
            far.append(round((num/n_non_comp) * 100, 2))

        far = np.array(far) 
        return far


    def __calculate_frr(self, y_true, y_pred):
        frr = []
        n_comp = len([x for x in y_true if x == Eval.COMPLIANT.value])
        th_range = np.arange(self.THRESH_START_VAL, self.THRESH_END_VAL, self.THRESH_STEP_SIZE) 
        for th in th_range:
            num = 0
            for tr_val,pred in zip(y_true,y_pred):
                if pred < th and tr_val == Eval.COMPLIANT.value:
                    num += 1
            frr.append(round((num/n_comp) * 100, 2))

        frr = np.array(frr)    
        return frr
    

    def __draw_far_frr_curve(self, th_range, frr, far, eer, req, best_th):
        fig = plt.figure(1)
        plt.plot(th_range, frr,'-r')
        plt.plot(th_range, far,'-b')
        plt.scatter(best_th, round(eer*100,4), marker='^', color='green', label='EER', s=70.)
        plt.xlabel('Threshold')
        plt.ylabel('FAR/FRR %')
        plt.xlim([0, 1.02])
        plt.ylim([0, 100])
        plt.title(f'Req: {req.upper()} - EER = {round(eer,4)} - {self.data_src.value.upper()}')
        plt.legend(['FRR','FAR'], loc='upper center')
        plt.show()
        
        if self.use_neptune:
            self.neptune_run[f'{self.vis_var_base_path}/far_frr_curve.png'].upload(fig)


    def __draw_roc_curve(self, fpr, tpr, eer, th, req):
        fig = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Req: {} | EER: {:.4f} | Thresh: {:.4f} | {}'.format(req.upper(), eer, th, self.data_src.value.upper()))
        plt.show()
        
        if self.use_neptune:
            self.neptune_run[f'{self.vis_var_base_path}/roc_curve.png'].upload(fig)
    
    
    def calculate_eer(self, req, verbose, running_nas):
        if self.y_test_true is None or self.y_test_hat is None:
            raise Exception('Call method make_predictions() before calculate_eer()!')
        
        y_true = self.y_test_true
        y_pred = self.y_test_hat
        
        fpr, tpr, ths = roc_curve(y_true, y_pred)
        far = self.__calculate_far(y_true, y_pred)
        frr = self.__calculate_frr(y_true, y_pred)
        th_range = np.arange(self.THRESH_START_VAL, self.THRESH_END_VAL, self.THRESH_STEP_SIZE) 
        
        EER_interp = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        best_th = interp1d(fpr, ths)(EER_interp)
        
        if verbose:
            self.__draw_roc_curve(fpr, tpr, EER_interp, best_th, req)
            self.__draw_far_frr_curve(th_range, frr, far, EER_interp, req, best_th)

        best_th = best_th.tolist()
        EER_interp = round(EER_interp, 4)

        if verbose:
            print(f'Requisite: {req.upper()} - EER_interp: {EER_interp*100}% - Best Threshold: {best_th}')

        self.y_test_hat_discrete = np.where(self.y_test_hat < best_th, 0, 1)
        
        if self.use_neptune and not running_nas:
            if not self.is_mtl_model:
                self.neptune_run[f'{self.metrics_var_base_path}/EER_interp'] = EER_interp
                self.neptune_run[f'{self.metrics_var_base_path}/best_th'] = best_th
            else:
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/EER_interp'] = EER_interp
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/best_th'] = best_th
        
        return EER_interp, best_th


    def get_classification_report(self, data_gen, verbose):
        if self.y_test_true is None or self.y_test_hat_discrete is None:
            raise Exception('Call method make_predictions() and calculate_eer() before __get_classification_report()!')
        
        target_names,labels = None,None
        if self.is_mtl_model:
            target_names = [Eval.NON_COMPLIANT.name, Eval.COMPLIANT.name]
            labels = [Eval.NON_COMPLIANT.value, Eval.COMPLIANT.value]
        else:
            target_names = list(data_gen.class_indices.keys())
            labels = list(data_gen.class_indices.values())
        
        if verbose:
            print('Classification report -----------------------------------')
            print(classification_report(y_true=self.y_test_true, 
                                        y_pred=self.y_test_hat_discrete, 
                                        target_names=target_names, 
                                        labels=labels))

    def calculate_accuracy(self, req, verbose, runnning_nas):
        if self.y_test_true is None or self.y_test_hat_discrete is None:
            raise Exception('Call method make_predictions() and calculate_eer() before calculate_accuracy()!')
        
        ACC = round(accuracy_score(self.y_test_true, self.y_test_hat_discrete), 4)
        
        if verbose:
            print('Accuracy ------------------------------------------------')
            print(f'Model Accuracy: {ACC*100}%')
            print('---------------------------------------------------------')
        
        if self.use_neptune and not runnning_nas:
            if not self.is_mtl_model:
                self.neptune_run[f'{self.metrics_var_base_path}/ACC'] = ACC
            else:
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/ACC'] = ACC
        
        return ACC


    def get_confusion_matrix(self, req, verbose, running_nas):
        if self.y_test_true is None or self.y_test_hat is None:
            raise Exception('Call method make_predictions() before calculate_confusion_matrix()!')
        
        TN,FP,FN,TP = confusion_matrix(self.y_test_true, 
                                       self.y_test_hat_discrete, 
                                       labels=[Eval.NON_COMPLIANT.value, Eval.COMPLIANT.value]).ravel()
        FAR = round(FP/(FP+TN),4)
        FRR = round(FN/(FN+TP),4)
        EER_mean = round((FAR+FRR)/2.,4)
        
        if verbose:
            print('Confusion matrix ----------------------------------------')
            print(f'FAR: {FAR*100}% | FRR: {FRR*100}% | EER_mean: {EER_mean*100}% | TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}')
        
        if self.use_neptune and not running_nas:
            if not self.is_mtl_model:
                self.neptune_run[f'{self.metrics_var_base_path}/TP'] = TP
                self.neptune_run[f'{self.metrics_var_base_path}/TN'] = TN
                self.neptune_run[f'{self.metrics_var_base_path}/FP'] = FP
                self.neptune_run[f'{self.metrics_var_base_path}/FN'] = FN
                self.neptune_run[f'{self.metrics_var_base_path}/FAR'] = FAR
                self.neptune_run[f'{self.metrics_var_base_path}/FRR'] = FRR
                self.neptune_run[f'{self.metrics_var_base_path}/EER_mean'] = EER_mean
            else:
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/TP'] = TP
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/TN'] = TN
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/FP'] = FP
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/FN'] = FN
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/FAR'] = FAR
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/FRR'] = FRR
                self.neptune_run[f'{self.metrics_var_base_path}/{req}/EER_mean'] = EER_mean
        
        return TN,TP,FN,FP,FAR,FRR,EER_mean

    
    def __calculate_metrics(self, predIdxs, data_gen, req, verbose, running_nas):
        self.y_test_hat = np.array([y1 for (_,y1) in predIdxs])  # COMPLIANT label predictions (class==1.0) (positive class)
        
        req_eval = RequisiteEvaluation(req)

        req = req.value.lower()
        
        EER_interp, best_thresh = self.calculate_eer(req, verbose, running_nas)
        req_eval.EER_interp = EER_interp
        req_eval.best_th = best_thresh
        
        self.get_classification_report(data_gen, verbose)

        TN,TP,FN,FP,FAR,FRR,EER_mean = self.get_confusion_matrix(req, verbose, running_nas)
        req_eval.TP = TP
        req_eval.TN = TN
        req_eval.FP = FP
        req_eval.FN = FN
        req_eval.FAR = FAR
        req_eval.FRR = FRR
        req_eval.EER_mean = EER_mean

        acc = self.calculate_accuracy(req, verbose, running_nas)
        req_eval.ACC = acc

        return req_eval
        
        
    
    def test_model(self, data_gen, model, verbose=True, running_nas=False):
        print("Testing Trained Model")
        
        print('Predicting labels....')
        data_gen.reset()
        predIdxs = model.predict(data_gen, batch_size=self.net_args['batch_size'], verbose=1)
        print('Prediction finished!')
        
        req_evaluations = []
        if self.is_mtl_model:
            for idx,req in enumerate(self.prop_args['reqs']):
                if req == ICAO_REQ.INK_MARK:    # TODO corrigir esse problema!!
                    continue
                print(f'Requisite: {req.value.upper()}') if verbose else None
                self.y_test_true = np.array(data_gen.labels[idx])
                req_evaluations.append(self.__calculate_metrics(predIdxs[idx], data_gen, req, verbose, running_nas))
        else:
            print(f'Requisite: {self.prop_args["reqs"][0].value.upper()}') if verbose else None
            self.y_test_true = np.array(data_gen.labels)
            req = self.prop_args['reqs'][0]
            req_evaluations.append(self.__calculate_metrics(predIdxs, data_gen, req, verbose, running_nas))
        
        final_eval = FinalEvaluation(req_evaluations)
        final_metrics = final_eval.calculate_final_metrics()
        print(final_eval)

        if self.use_neptune and not running_nas:
            final_eval.log_to_neptune(self.neptune_run, self.data_src)

        return final_metrics
    
    
    # Calculates heatmaps of GradCAM algorithm based on the following implementations:
    ## https://stackoverflow.com/questions/58322147/how-to-generate-cnn-heatmaps-using-built-in-keras-in-tf2-0-tf-keras 
    ## https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-gradcam-554a85dd4e48
    def __calc_heatmap(self, img_name, base_model, model):
        image = load_img(img_name, target_size=base_model.value['target_size'])
        img_tensor = img_to_array(image)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = base_model.value['prep_function'](img_tensor)

        last_conv_layer_name = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]

        conv_layer = model.get_layer(last_conv_layer_name)
        heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

        # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
        with tf.GradientTape() as gtape:
            conv_output, predictions = heatmap_model(img_tensor)
            loss = predictions[:, np.argmax(predictions[0])]
            grads = gtape.gradient(loss, conv_output)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # Channel-wise mean of resulting feature-map is the heatmap of class activation
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat

        # Render heatmap via pyplot
        # plt.imshow(heatmap[0])
        # plt.show()

        upsample = cv2.resize(heatmap[0], base_model.value['target_size'])
        return upsample
        
    
    def __log_imgs_sample(self, data_df, data_pred_selection):
        if self.use_neptune:
            print(f"Logging sample of {data_pred_selection.value['abv'].upper()} images to Neptune")
            for index, row in data_df.iterrows():
                self.neptune_run[f"images/{self.data_src.value.lower()}/{data_pred_selection.value['abv']}"].log(File(row['img_name']))
        else:
            print('Not using Neptune!')
    
    
    def __select_viz_data(self, data_gen, preds, n_imgs, data_pred_selection):
        tmp_df = pd.DataFrame()
        tmp_df['img_name'] = data_gen.filepaths
        tmp_df['comp'] = data_gen.labels
        tmp_df['pred'] = preds
        
        data_src_uppercase = self.data_src.value.upper()
        data_src_lowercase = self.data_src.value.lower()
        
        viz_title, neptune_viz_path = None, None
        if data_pred_selection.name == DataPredSelection.ONLY_FN.name:
            tmp_df = tmp_df[(tmp_df.comp == Eval.COMPLIANT.value) & (tmp_df.pred == Eval.NON_COMPLIANT.value)]
        elif data_pred_selection.name == DataPredSelection.ONLY_FP.name:
            tmp_df = tmp_df[(tmp_df.comp == Eval.NON_COMPLIANT.value) & (tmp_df.pred == Eval.COMPLIANT.value)]
        elif data_pred_selection.name == DataPredSelection.ONLY_TP.name:
            tmp_df = tmp_df[(tmp_df.comp == Eval.COMPLIANT.value) & (tmp_df.pred == Eval.COMPLIANT.value)]
        elif data_pred_selection.name == DataPredSelection.ONLY_TN.name:
            tmp_df = tmp_df[(tmp_df.comp == Eval.NON_COMPLIANT.value) & (tmp_df.pred == Eval.NON_COMPLIANT.value)]
        
        n_imgs = tmp_df.shape[0] if tmp_df.shape[0] < n_imgs else n_imgs
        tmp_df = tmp_df.sample(n=n_imgs, random_state=SEED)
        
        #self.__log_imgs_sample(tmp_df, data_pred_selection)
        
        viz_title = f"{data_pred_selection.value['title']} - {self.data_src.value.upper()}" 
        neptune_viz_path = f"{self.vis_var_base_path}/predictions_with_heatmaps/{data_pred_selection.value['abv']}"
        
        return tmp_df, viz_title, neptune_viz_path
    
    
    def __get_img_name(self, img_path):
            return img_path.split("/")[-1].split(".")[0]
    
    # sort 50 samples from test_df, calculates GradCAM heatmaps
    # and log the resulting images in a grid to neptune
    def visualize_predictions(self, base_model, model, data_gen, n_imgs, data_pred_selection):
        preds = self.y_test_hat_discrete
        tmp_df,viz_title, neptune_viz_path = self.__select_viz_data(data_gen, preds, n_imgs, data_pred_selection)
        
        labels = [f'GT: COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'GT: NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.comp.values, tmp_df.img_name.values)]
        
        preds = [f'GT: COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'GT: NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.pred.values, tmp_df.img_name.values)]
        
        heatmaps = [self.__calc_heatmap(im_name, base_model, model) for im_name in tmp_df.img_name.values]
        
        imgs = [cv2.resize(cv2.imread(img), base_model.value['target_size']) for img in tmp_df.img_name.values]
        
        f = dr.draw_imgs(imgs, title=viz_title, labels=labels, predictions=preds, heatmaps=heatmaps)
        
        if self.use_neptune and f is not None:
            self.neptune_run[neptune_viz_path].upload(f)
    