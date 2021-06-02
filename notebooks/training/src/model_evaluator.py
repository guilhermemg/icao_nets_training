import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from enum import Enum

import matplotlib.pyplot as plt

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
        
        self.metrics_var_base_path = None  # base path of metrics-variables in Neptune
        self.viz_var_base_path = None # base path of vizualizations-variables in Neptune
        
        self.THRESH_START_VAL = 0.0
        self.THRESH_END_VAL = 1.02
        self.THRESH_STEP_SIZE = 1e-2
    
    
    def set_data_src(self, data_src):
        if data_src in list(DataSource):
            self.data_src = data_src
            self.metrics_var_base_path = f'metrics/{self.data_src.value}'
            self.viz_var_base_path = f'viz/{self.data_src.value}'
        else:
            raise Exception(f'Error! Invalid data source. Valid Options: {list(DataSource)}')
    
    
    def __draw_roc_curve(self, fpr, tpr, eer, th, req, data_src):
        fig = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Req: {} | EER: {:.4f} | Thresh: {:.4f} | {}'.format(req.value.upper(), eer, th, data_src.upper()))
        plt.show()
        return fig


    def __draw_far_frr_curve(self, th_range, frr, far, eer, req_name, data_src):
        fig = plt.figure(1)
        plt.plot(th_range, frr,'-r')
        plt.plot(th_range, far,'-b')
        plt.xlabel('Threshold')
        plt.ylabel('FAR/FRR %')
        plt.xlim([0, 1.02])
        plt.ylim([0, 100])
        plt.title(f'Req: {req_name.value.upper()} - EER = {round(eer,4)} - {data_src.upper()}')
        plt.legend(['FRR','FAR'], loc='upper center')
        plt.show()
        return fig


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


    def calculate_eer(self, y_true, y_pred, req, data_src):
        fpr, tpr, ths = roc_curve(y_true, y_pred)
        far = self.__calculate_far(y_true, y_pred)
        frr = self.__calculate_frr(y_true, y_pred)
        th_range = np.arange(self.THRESH_START_VAL, self.THRESH_END_VAL, self.THRESH_STEP_SIZE) 
        
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        best_th = interp1d(fpr, ths)(eer)
        
        #print(f'ths: {ths}')
        #print(f'fpr: {fpr[:20]}')
        #print(f'tpr: {tpr[:20]}')
        #print(f'FPRxTPR: eer: {eer} | best_th: {best_th}')
        
        #print(f'far: {far[:20]}')
        #print(f'frr: {frr[:20]}')
        #eer2 = brentq(lambda x : 1. - x - interp1d(far, frr)(x), 0., 1.)
        #best_th2 = interp1d(fpr, ths)(eer)
        #print(f'FARxFRR: eer: {eer2} | best_th: {best_th2}')
        
        #fnr = 1 - tpr
        #print(np.absolute((fnr - fpr)))
        #print(np.nanargmin(np.absolute((fnr - fpr))))
        #eer_threshold = ths[np.nanargmin(np.absolute((fnr - fpr)))]
        #EER_2 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        #EER_3 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        #print(f'thresh: {eer_threshold} | EER_2: {EER_2} | EER_3: {EER_3}')

        roc_curve_fig = self.__draw_roc_curve(fpr, tpr, eer, best_th, req, data_src)
        far_frr_curve_fig = self.__draw_far_frr_curve(th_range=th_range, far=far, frr=frr, eer=eer, req_name=req, data_src=data_src)

        best_th = best_th.tolist()
        eer = round(eer, 4)
        print(f'Requisite: {req} - EER: {eer*100}% - Best Threshold: {best_th}')

        return eer, best_th, roc_curve_fig, far_frr_curve_fig


    def get_classification_report(self, test_gen, y_true, y_pred, req_name):
        print('Classification report -----------------------------------')
        
        target_names,labels = None,None
        if self.is_mtl_model:
            target_names = [Eval.NON_COMPLIANT.name, Eval.COMPLIANT.name]
            labels = [Eval.NON_COMPLIANT.value, Eval.COMPLIANT.value]
        else:
            target_names = list(test_gen.class_indices.keys())
            labels = list(test_gen.class_indices.values())
        
        print(classification_report(y_true=y_true, 
                                    y_pred=y_pred, 
                                    target_names=target_names, 
                                    labels=labels))

    def calculate_accuracy(self, y_true, y_pred):
        print('Accuracy ------------------------------------------------')
        acc = round(accuracy_score(y_true, y_pred), 4)
        print(f'Model Accuracy: {acc*100}%')
        print('---------------------------------------------------------')
        return acc


    def get_confusion_matrix(self, y_true, y_pred):
        print('Confusion matrix ----------------------------------------')
        TN,FP,FN,TP = confusion_matrix(y_true, y_pred, labels=[Eval.NON_COMPLIANT.value, Eval.COMPLIANT.value]).ravel()
        FAR = round(FP/(FP+TN),4)
        FRR = round(FN/(FN+TP),4)
        print(f'FAR: {FAR*100}% | FRR: {FRR*100}% | TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}')
        return FAR,FRR,TN,FP,FN,TP

    
    def __log_test_metrics(self, predIdxs, req_name, test_gen, data_src):
        self.y_test_hat = np.array([y1 for (_,y1) in predIdxs])  # COMPLIANT label predictions (class==1.0) (positive class)
        
        eer,best_th,roc_curve_fig,far_frr_curve_fig = self.calculate_eer(self.y_test_true, self.y_test_hat, req_name, data_src)
        
        self.y_test_hat_discrete = np.where(self.y_test_hat < best_th, 0, 1)
        
        self.get_classification_report(test_gen, self.y_test_true, self.y_test_hat_discrete, req_name)
        acc = self.calculate_accuracy(self.y_test_true, self.y_test_hat_discrete)
        FAR,FRR,TN,FP,FN,TP = self.get_confusion_matrix(self.y_test_true, self.y_test_hat_discrete)
        
        if self.use_neptune:
            self.neptune_run[f'viz/{data_src}/roc_curve'].upload(roc_curve_fig)
            self.neptune_run[f'viz/{data_src}/far_frr_curve'].upload(far_frr_curve_fig)
            self.neptune_run[f'metrics/{data_src}/eer'] = eer
            self.neptune_run[f'metrics/{data_src}/best_th'] = best_th
            self.neptune_run[f'metrics/{data_src}/TP'] = TP
            self.neptune_run[f'metrics/{data_src}/TN'] = TN
            self.neptune_run[f'metrics/{data_src}/FP'] = FP
            self.neptune_run[f'metrics/{data_src}/FN'] = FN
            self.neptune_run[f'metrics/{data_src}/FAR'] = FAR
            self.neptune_run[f'metrics/{data_src}/FRR'] = FRR
            self.neptune_run[f'metrics/{data_src}/eval_acc'] = acc
    
    
    def test_model(self, data_src, data_gen, model, is_mtl_model):
        print("Testing Trained Model")
        
        print('Predicting labels....')
        data_gen.reset()
        predIdxs = model.predict(data_gen, batch_size=self.net_args['batch_size'], verbose=1)
        print('Prediction finished!')
        
        if is_mtl_model:
            for idx,req in enumerate(self.prop_args['reqs']):
                #if req == cts.ICAO_REQ.INK_MARK:    # TODO corrigir esse problema!!
                #    continue
                print(f'Requisite: {req.value.upper()}')
                
                self.y_test_true = np.array(data_gen.labels[idx])
                
                self.__log_test_metrics(predIdxs[idx], req, data_gen, data_src)
        else:
            print(f'Requisite: {self.prop_args["reqs"][0].value.upper()}')
            
            self.y_test_true = np.array(data_gen.labels)
            
            self.__log_test_metrics(predIdxs, self.prop_args['reqs'][0], data_gen, data_src)
    
    
#     def evaluate_model(self, data_gen, model):
#         print('Evaluating model')
#         eval_metrics = model.evaluate(data_gen, verbose=0)
        
#         print(f'Loss: {round(eval_metrics[0], 4)}')
#         print(f'Accuracy: {round(eval_metrics[1]*100, 2)}%')
        
#         if self.use_neptune:
#             for j, metric in enumerate(eval_metrics):
#                 neptune.log_metric('eval_' + model.metrics_names[j], metric)
    
    
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
    
    
    def __select_viz_data(self, data_src, data_gen, preds, n_imgs, show_only_fp, show_only_fn, show_only_tp, show_only_tn):
        tmp_df = pd.DataFrame()
        tmp_df['img_name'] = data_gen.filepaths
        tmp_df['comp'] = data_gen.labels
        tmp_df['pred'] = preds
        
        viz_title, neptune_viz_path = None, None
        if show_only_fn:
            tmp_df = tmp_df[(tmp_df.comp == Eval.COMPLIANT.value) & (tmp_df.pred == Eval.NON_COMPLIANT.value)]
            viz_title = f"Only False Negatives images - {data_src.upper()}" 
            neptune_viz_path = f'viz/{data_src}/predictions_with_heatmaps_fn_only'
        elif show_only_fp:
            tmp_df = tmp_df[(tmp_df.comp == Eval.NON_COMPLIANT.value) & (tmp_df.pred == Eval.COMPLIANT.value)]
            viz_title = f"Only False Positive images - {data_src.upper()}" 
            neptune_viz_path = f'viz/{data_src}/predictions_with_heatmaps_fp_only'
        elif show_only_tp:
            tmp_df = tmp_df[(tmp_df.comp == Eval.COMPLIANT.value) & (tmp_df.pred == Eval.COMPLIANT.value)]
            viz_title = f"Only True Positive images - {data_src.upper()}" 
            neptune_viz_path = f'viz/{data_src}/predictions_with_heatmaps_tp_only'
        elif show_only_tn:
            tmp_df = tmp_df[(tmp_df.comp == Eval.NON_COMPLIANT.value) & (tmp_df.pred == Eval.NON_COMPLIANT.value)]
            viz_title = f"Only True Negatives images - {data_src.upper()}" 
            neptune_viz_path = f'viz/{data_src}/predictions_with_heatmaps_tn_only'
        else:
            viz_title = f"Any (TP,FP,TN,FN) images - {data_src.upper()}"
            neptune_viz_path = f'viz/{data_src}/predictions_with_heatmaps_any_img'
        
        n_imgs = tmp_df.shape[0] if tmp_df.shape[0] < n_imgs else n_imgs
        
        tmp_df = tmp_df.sample(n=n_imgs, random_state=SEED)
        
        return tmp_df, viz_title, neptune_viz_path
    
    
    def __get_img_name(self, img_path):
            return img_path.split("/")[-1].split(".")[0]
    
    # sort 50 samples from test_df, calculates GradCAM heatmaps
    # and log the resulting images in a grid to neptune
    def vizualize_predictions(self, data_src, base_model, model, test_gen, n_imgs, show_only_fp, show_only_fn, show_only_tp, show_only_tn):
        preds = self.y_test_hat_discrete
        tmp_df,viz_title, neptune_viz_path = self.__select_viz_data(data_src, test_gen, preds, n_imgs, show_only_fp, show_only_fn,
                                                                       show_only_tp, show_only_tn)
        
        
        labels = [f'GT: COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'GT: NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.comp.values, tmp_df.img_name.values)]
        
        preds = [f'GT: COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'GT: NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.pred.values, tmp_df.img_name.values)]
        
        heatmaps = [self.__calc_heatmap(im_name, base_model, model) for im_name in tmp_df.img_name.values]
        
        imgs = [cv2.resize(cv2.imread(img), base_model.value['target_size']) for img in tmp_df.img_name.values]
        
        f = dr.draw_imgs(imgs, title=viz_title, labels=labels, predictions=preds, heatmaps=heatmaps)
        
        if self.use_neptune:
            self.neptune_run[neptune_viz_path].upload(f)
    