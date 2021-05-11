import cv2
import neptune
import numpy as np
import pandas as pd
import tensorflow as tf

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

class ModelEvaluator:
    def __init__(self, net_args, prop_args, is_mtl_model, use_neptune, script_mode):
        self.net_args = net_args
        self.prop_args = prop_args
        self.is_mtl_model = is_mtl_model
        self.use_neptune = use_neptune
        self.script_mode = script_mode
    
    
    def __draw_roc_curve(self, fpr, tpr, eer, th, req):
        fig = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve - Req: {} | EER: {:.4f} | Thresh: {:.4f}'.format(req.value.upper(), eer, th))
        if not self.script_mode:
            plt.show()
        return fig


    def __draw_far_frr_curve(self, th_range, frr, far, eer, req_name):
        fig = plt.figure(1)
        plt.plot(th_range, frr,'-r')
        plt.plot(th_range, far,'-b')
        plt.xlabel('Threshold')
        plt.ylabel('FAR/FRR %')
        plt.xlim([0, 1.02])
        plt.ylim([0, 100])
        plt.title(f'Req: {req_name.value.upper()} - EER = {round(eer,4)}')
        plt.legend(['FRR','FAR'], loc='upper center')
        if not self.script_mode:
            plt.show()
        return fig


    def __calculate_far(self, y_true, y_pred):
        far = []
        n_attacks = len([x for x in y_true if x == Eval.NON_COMPLIANT.value])
        th_range = np.arange(0, 1.02, 0.01) 
        for th in th_range:
            num = 0
            for tr_val,pred in zip(y_true,y_pred):
                if pred >= th and tr_val == Eval.NON_COMPLIANT.value:
                    num += 1
            far.append(round((num/n_attacks) * 100, 2))

        far = np.array(far) 
        return far


    def __calculate_frr(self, y_true, y_pred):
        frr = []
        n_reals = len([x for x in y_true if x == Eval.COMPLIANT.value])
        th_range = np.arange(0, 1.02, 0.01) 
        for th in th_range:
            num = 0
            for tr_val,pred in zip(y_true,y_pred):
                if pred < th and tr_val == Eval.COMPLIANT.value:
                    num += 1
            frr.append(round((num/n_reals) * 100, 2))

        frr = np.array(frr)    
        return frr


    def calculate_eer(self, y_true, y_pred, req):
        fpr, tpr, ths = roc_curve(y_true, y_pred)
        far = self.__calculate_far(y_true, y_pred)
        frr = self.__calculate_frr(y_true, y_pred)
        th_range = np.arange(0, 1.02, 0.01)

        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        best_th = interp1d(fpr, ths)(eer)

        roc_curve_fig = self.__draw_roc_curve(fpr, tpr, eer, best_th, req)
        far_frr_curve_fig = self.__draw_far_frr_curve(th_range=th_range, far=far, frr=frr, eer=eer, req_name=req)

        best_th = round(best_th.tolist(), 4)
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

    
    def __log_test_metrics(self, predIdxs, req_name, test_gen):
        self.y_test_hat = np.array([y1 for (_,y1) in predIdxs])  # COMPLIANT label predictions (class==1.0) (positive class)
        
        eer,best_th,roc_curve_fig,far_frr_curve_fig = self.calculate_eer(self.y_test_true, self.y_test_hat, req_name)
        
        self.y_test_hat_discrete = np.where(self.y_test_hat < best_th, 0, 1)
        
        self.get_classification_report(test_gen, self.y_test_true, self.y_test_hat_discrete, req_name)
        acc = self.calculate_accuracy(self.y_test_true, self.y_test_hat_discrete)
        FAR,FRR,TN,FP,FN,TP = self.get_confusion_matrix(self.y_test_true, self.y_test_hat_discrete)

        if self.use_neptune:
            neptune.send_image('roc_curve.png', roc_curve_fig)
            neptune.send_image('far_frr_curve.png', far_frr_curve_fig)
            neptune.log_metric('eer', eer)
            neptune.log_metric('best_th', best_th)
            neptune.log_metric('TP', TP)
            neptune.log_metric('TN', TN)
            neptune.log_metric('FP', FP)
            neptune.log_metric('FN', FN)
            neptune.log_metric('FAR', FAR)
            neptune.log_metric('FRR', FRR)
            neptune.log_metric('eval_acc', acc)
    
    
    def test_model(self, test_gen, model, is_mtl_model):
        print("Testing Trained Model")
        
        print('Predicting labels....')
        test_gen.reset()
        predIdxs = model.predict(test_gen, batch_size=self.net_args['batch_size'], verbose=1)
        print('Prediction finished!')
        
        if is_mtl_model:
            for idx,req in enumerate(self.prop_args['reqs']):
                if req == cts.ICAO_REQ.INK_MARK:    # TODO corrigir esse problema!!
                    continue
                print(f'Requisite: {req.value.upper()}')
                
                self.y_test_true = np.array(test_gen.labels[idx])
                
                self.__log_test_metrics(predIdxs[idx], req, test_gen)
        else:
            print(f'Requisite: {self.prop_args["reqs"][0].value.upper()}')
            
            self.y_test_true = np.array(test_gen.labels)
            
            self.__log_test_metrics(predIdxs, self.prop_args['reqs'][0], test_gen)
    
    
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
    
    
    def __select_viz_data(self, data_gen, preds, n_imgs, show_only_fp, show_only_fn, show_only_tp, show_only_tn):
        tmp_df = pd.DataFrame()
        tmp_df['img_name'] = data_gen.filepaths
        tmp_df['comp'] = data_gen.labels
        tmp_df['pred'] = preds
        
        viz_title = None
        if show_only_fn:
            tmp_df = tmp_df[(tmp_df.comp == Eval.COMPLIANT.value) & (tmp_df.pred == Eval.NON_COMPLIANT.value)]
            viz_title = "Only False Negatives images" 
        elif show_only_fp:
            tmp_df = tmp_df[(tmp_df.comp == Eval.NON_COMPLIANT.value) & (tmp_df.pred == Eval.COMPLIANT.value)]
            viz_title = "Only False Positive images" 
        elif show_only_tp:
            tmp_df = tmp_df[(tmp_df.comp == Eval.COMPLIANT.value) & (tmp_df.pred == Eval.COMPLIANT.value)]
            viz_title = "Only True Positive images" 
        elif show_only_tn:
            tmp_df = tmp_df[(tmp_df.comp == Eval.NON_COMPLIANT.value) & (tmp_df.pred == Eval.NON_COMPLIANT.value)]
            viz_title = "Only True Negatives images" 
        else:
            viz_title = "Any (TP,FP,TN,FN) images"
        
        n_imgs = tmp_df.shape[0] if tmp_df.shape[0] < n_imgs else n_imgs
        
        tmp_df = tmp_df.sample(n=n_imgs, random_state=SEED)
        
        return tmp_df, viz_title
    
    
    def __get_img_name(self, img_path):
            return img_path.split("/")[-1].split(".")[0]
    
    # sort 50 samples from test_df, calculates GradCAM heatmaps
    # and log the resulting images in a grid to neptune
    def vizualize_predictions(self, base_model, model, test_gen, n_imgs, show_only_fp, show_only_fn, show_only_tp, show_only_tn):
        preds = self.y_test_hat_discrete
        tmp_df,viz_title = self.__select_viz_data(test_gen, preds, n_imgs, show_only_fp, show_only_fn,
                                       show_only_tp, show_only_tn)
        
        
        labels = [f'COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.comp.values, tmp_df.img_name.values)]
        
        preds = [f'COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.pred.values, tmp_df.img_name.values)]
        
        heatmaps = [self.__calc_heatmap(im_name, base_model, model) for im_name in tmp_df.img_name.values]
        
        imgs = [cv2.resize(cv2.imread(img), base_model.value['target_size']) for img in tmp_df.img_name.values]
        
        f = dr.draw_imgs(imgs, labels=labels, predictions=preds, heatmaps=heatmaps, script_mode=self.script_mode)
        
        if self.use_neptune:
            neptune.send_image('predictions_with_heatmaps.png',f)
    