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

import utils.constants as cts
import utils.draw_utils as dr

from utils.constants import SEED


class DataSource(Enum):
    VALIDATION = 'validation'
    TEST = 'test'  
    

class ModelEvaluator:
    def __init__(self, net_args, prop_args, is_mtl_model, neptune_run):
        self.net_args = net_args
        self.prop_args = prop_args
        self.is_mtl_model = is_mtl_model
        self.neptune_run = neptune_run
        self.use_neptune = True if neptune_run is not None else False
        
        self.data_src = None
        
        self.metrics_var_base_path = None  # base path of metrics-variables in Neptune
        self.viz_var_base_path = None # base path of vizualizations-variables in Neptune
        
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
            self.viz_var_base_path = f'viz/{self.data_src.value.lower()}'
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
            self.neptune_run[f'{self.viz_var_base_path}/{req}/far_frr_curve.png'].upload(fig)


    def __draw_roc_curve(self, fpr, tpr, eer, th, req):
        fig = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Req: {} | EER: {:.4f} | Thresh: {:.4f} | {}'.format(req.upper(), eer, th, self.data_src.value.upper()))
        plt.show()
        
        if self.use_neptune:
            self.neptune_run[f'{self.viz_var_base_path}/{req}/roc_curve.png'].upload(fig)
    
    
    def calculate_eer(self, req):
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
        
        self.__draw_roc_curve(fpr, tpr, EER_interp, best_th, req)
        self.__draw_far_frr_curve(th_range, frr, far, EER_interp, req, best_th)

        best_th = best_th.tolist()
        EER_interp = round(EER_interp, 4)
        print(f'Requisite: {req.upper()} - EER_interp: {EER_interp*100}% - Best Threshold: {best_th}')

        self.y_test_hat_discrete = np.where(self.y_test_hat < best_th, 0, 1)
            
        if self.use_neptune:
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/EER_interp'] = EER_interp
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/best_th'] = best_th


    def get_classification_report(self, data_gen):
        if self.y_test_true is None or self.y_test_hat_discrete is None:
            raise Exception('Call method make_predictions() and calculate_eer() before __get_classification_report()!')
        
        print('Classification report -----------------------------------')
        
        target_names,labels = None,None
        if self.is_mtl_model:
            target_names = [Eval.NON_COMPLIANT.name, Eval.COMPLIANT.name]
            labels = [Eval.NON_COMPLIANT.value, Eval.COMPLIANT.value]
        else:
            target_names = list(data_gen.class_indices.keys())
            labels = list(data_gen.class_indices.values())
        
        print(classification_report(y_true=self.y_test_true, 
                                    y_pred=self.y_test_hat_discrete, 
                                    target_names=target_names, 
                                    labels=labels))

    def calculate_accuracy(self, req):
        if self.y_test_true is None or self.y_test_hat_discrete is None:
            raise Exception('Call method make_predictions() and calculate_eer() before calculate_accuracy()!')
        
        print('Accuracy ------------------------------------------------')
        ACC = round(accuracy_score(self.y_test_true, self.y_test_hat_discrete), 4)
        print(f'Model Accuracy: {ACC*100}%')
        print('---------------------------------------------------------')
        
        if self.use_neptune:
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/ACC'] = ACC


    def get_confusion_matrix(self, req):
        if self.y_test_true is None or self.y_test_hat is None:
            raise Exception('Call method make_predictions() before calculate_confusion_matrix()!')
        
        print('Confusion matrix ----------------------------------------')
        TN,FP,FN,TP = confusion_matrix(self.y_test_true, 
                                       self.y_test_hat_discrete, 
                                       labels=[Eval.NON_COMPLIANT.value, Eval.COMPLIANT.value]).ravel()
        FAR = round(FP/(FP+TN),4)
        FRR = round(FN/(FN+TP),4)
        EER_mean = round((FAR+FRR)/2.,4)
        
        print(f'FAR: {FAR*100}% | FRR: {FRR*100}% | EER_mean: {EER_mean*100}% | TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}')
        
        if self.use_neptune:
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/TP'] = TP
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/TN'] = TN
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/FP'] = FP
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/FN'] = FN
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/FAR'] = FAR
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/FRR'] = FRR
            self.neptune_run[f'{self.metrics_var_base_path}/{req}/EER_mean'] = EER_mean

    
    def __calculate_metrics(self, predIdxs, data_gen, req):
        self.y_test_hat = np.array([y1 for (_,y1) in predIdxs])  # COMPLIANT label predictions (class==1.0) (positive class)
        
        req = req.value.lower()
        
        self.calculate_eer(req)
        self.get_classification_report(data_gen)
        self.get_confusion_matrix(req)
        self.calculate_accuracy(req)
        
        
    
    def test_model(self, data_gen, model):
        print("Testing Trained Model")
        
        print('Predicting labels....')
        data_gen.reset()
        predIdxs = model.predict(data_gen, batch_size=self.net_args['batch_size'], verbose=1)
        print('Prediction finished!')
        
        if self.is_mtl_model:
            for idx,req in enumerate(self.prop_args['reqs']):
                #if req == cts.ICAO_REQ.INK_MARK:    # TODO corrigir esse problema!!
                #    continue
                print(f'Requisite: {req.value.upper()}')
                self.y_test_true = np.array(data_gen.labels[idx])
                self.__calculate_metrics(predIdxs[idx], data_gen, req)
        else:
            print(f'Requisite: {self.prop_args["reqs"][0].value.upper()}')
            self.y_test_true = np.array(data_gen.labels)
            self.__calculate_metrics(predIdxs, data_gen, self.prop_args['reqs'][0])
    
    
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
    
    
    def __select_viz_data(self, data_gen, preds, n_imgs, show_only_fp, show_only_fn, show_only_tp, show_only_tn):
        tmp_df = pd.DataFrame()
        tmp_df['img_name'] = data_gen.filepaths
        tmp_df['comp'] = data_gen.labels
        tmp_df['pred'] = preds
        partition = ''
        
        data_src_uppercase = self.data_src.value.upper()
        data_src_lowercase = self.data_src.value.lower()
        
        viz_title, neptune_viz_path = None, None
        if show_only_fn:
            tmp_df = tmp_df[(tmp_df.comp == Eval.COMPLIANT.value) & (tmp_df.pred == Eval.NON_COMPLIANT.value)]
            viz_title = f"Only False Negatives images - {data_src_uppercase}" 
            neptune_viz_path = f'viz/{data_src_lowercase}/predictions_with_heatmaps_fn_only'
            partition = 'fn'
        elif show_only_fp:
            tmp_df = tmp_df[(tmp_df.comp == Eval.NON_COMPLIANT.value) & (tmp_df.pred == Eval.COMPLIANT.value)]
            viz_title = f"Only False Positive images - {data_src_uppercase}" 
            neptune_viz_path = f'viz/{data_src_lowercase}/predictions_with_heatmaps_fp_only'
            partition = 'fp'
        elif show_only_tp:
            tmp_df = tmp_df[(tmp_df.comp == Eval.COMPLIANT.value) & (tmp_df.pred == Eval.COMPLIANT.value)]
            viz_title = f"Only True Positive images - {data_src_uppercase}" 
            neptune_viz_path = f'viz/{data_src_lowercase}/predictions_with_heatmaps_tp_only'
            partition = 'tp'
        elif show_only_tn:
            tmp_df = tmp_df[(tmp_df.comp == Eval.NON_COMPLIANT.value) & (tmp_df.pred == Eval.NON_COMPLIANT.value)]
            viz_title = f"Only True Negatives images - {data_src_uppercase}" 
            neptune_viz_path = f'viz/{data_src_lowercase}/predictions_with_heatmaps_tn_only'
            partition = 'tn'
        else:
            viz_title = f"Any (TP,FP,TN,FN) images - {data_src_uppercase}"
            neptune_viz_path = f'viz/{data_src_lowercase}/predictions_with_heatmaps_any_img'
            partition = 'all'
        
        n_imgs = tmp_df.shape[0] if tmp_df.shape[0] < n_imgs else n_imgs
        
        tmp_df = tmp_df.sample(n=n_imgs, random_state=SEED)
        
        if self.use_neptune:
            for index, row in tmp_df.iterrows():
                filename = row['img_name']
                self.neptune_run[str('sample_images/'+self.data_src.value.upper()+'/'+partition)].log(File(filename))
        
        return tmp_df, viz_title, neptune_viz_path
        
        return tmp_df, viz_title, neptune_viz_path
    
    
    def __get_img_name(self, img_path):
            return img_path.split("/")[-1].split(".")[0]
    
    # sort 50 samples from test_df, calculates GradCAM heatmaps
    # and log the resulting images in a grid to neptune
    def vizualize_predictions(self, base_model, model, data_gen, n_imgs, show_only_fp, show_only_fn, show_only_tp, show_only_tn):
        preds = self.y_test_hat_discrete
        tmp_df,viz_title, neptune_viz_path = self.__select_viz_data(data_gen, 
                                                                    preds, 
                                                                    n_imgs, 
                                                                    show_only_fp, 
                                                                    show_only_fn,
                                                                    show_only_tp, 
                                                                    show_only_tn)
        
        
        labels = [f'GT: COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'GT: NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.comp.values, tmp_df.img_name.values)]
        
        preds = [f'GT: COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'GT: NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.pred.values, tmp_df.img_name.values)]
        
        heatmaps = [self.__calc_heatmap(im_name, base_model, model) for im_name in tmp_df.img_name.values]
        
        imgs = [cv2.resize(cv2.imread(img), base_model.value['target_size']) for img in tmp_df.img_name.values]
        
        f = dr.draw_imgs(imgs, title=viz_title, labels=labels, predictions=preds, heatmaps=heatmaps)
        
        if self.use_neptune and f is not None:
            self.neptune_run[neptune_viz_path].upload(f)
    