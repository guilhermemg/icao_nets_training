import os
import sys
import cv2
import random
import datetime
import neptune
import tempfile
import numpy as np
import pandas as pd

from imutils import paths

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, InceptionV3
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_input_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as prep_input_inceptionv3
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if '../../../notebooks/' not in sys.path:
    sys.path.append('../../../notebooks/')

import utils.constants as cts
import utils.draw_utils as dr

from models.oface_mouth_model import OpenfaceMouth

from data_loaders.data_loader import DLName

from net_data_loaders.net_data_loader import NetDataLoader

from gt_loaders.gen_gt import Eval
from gt_loaders.fvc_gt import FVC_GTLoader
from gt_loaders.pybossa_gt import PybossaGTLoader


## restrict memory -------------------

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    raise Exception("Invalid device or cannot modify virtual devices once initialized.")

## restrict memory -------------------    

print('Starting Neptune')
neptune.init('guilhermemg/icao-nets-training')    
    
def log_data(logs):
    neptune.log_metric('epoch_accuracy', logs['accuracy'])
    neptune.log_metric('epoch_val_accuracy', logs['val_accuracy'])
    neptune.log_metric('epoch_loss', logs['loss'])    
    neptune.log_metric('epoch_val_loss', logs['val_loss'])    

    
def lr_scheduler(epoch):
    if epoch < 10:
        new_lr = PARAMS['learning_rate']
    else:
        new_lr = PARAMS['learning_rate'] * np.exp(0.1 * ((epoch//50)*50 - epoch))

    neptune.log_metric('learning_rate', new_lr)
    return new_lr

    
# Define parameters
PARAMS = {'batch_size': 32,
          'n_epochs': 40,
          'shuffle': True,
          'dense_units': 128,
          'learning_rate': 1e-4,
          'optimizer': 'Adam',
          'dropout': 0.5,
          'early_stopping':10
          }
    

m = OpenfaceMouth()
req = cts.ICAO_REQ.MOUTH

dl_names = [DLName.FVC_PYBOSSA, DLName.VGGFACE2, DLName.FEI_DB, DLName.GEORGIA_TECH,
           DLName.IMFDB, DLName.LFW, DLName.CELEBA]
print(f'DL names: {dl_names}')

print('Loading data')
netDataLoader = NetDataLoader(m, req, dl_names, True)
in_data = netDataLoader.load_data()
print('Data loaded')

print('Creating experiment')
neptune.create_experiment(name='train_mobilenetv2',
                          params=PARAMS,
                          properties={'dl_names': str([dl_name.value for dl_name in dl_names]),
                                      'dl_aligned': True,
                                      'icao_req': req.value,
                                      'tagger_model': m.get_model_name().value},
                          description='Increasing number of epochs from 10 to 40',
                          tags=['mobilenetv2'],
                          upload_source_files=['train_mobilenetv2.py'])

  
# # Network Training

# ## Training MobileNetV2

print('Starting data generators')
datagen = ImageDataGenerator(preprocessing_function=prep_input_mobilenetv2, 
                             validation_split=0.15,
                             rescale=1.0/255.0,
                             rotation_range=20,
                             zoom_range=0.15,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.15,
                             horizontal_flip=True,
                             fill_mode="nearest")

train_gen = datagen.flow_from_dataframe(in_data[:15000], 
                                        x_col="img_name", 
                                        y_col="comp",
                                        target_size=(224, 224),
                                        class_mode="binary",
                                        batch_size=32, 
                                        shuffle=True,
                                        subset='training',
                                        seed=0)

validation_gen = datagen.flow_from_dataframe(in_data[:15000],
                                            x_col="img_name", 
                                            y_col="comp",
                                            target_size=(224, 224),
                                            class_mode="binary",
                                            batch_size=32, 
                                            shuffle=True,
                                            subset='validation',
                                            seed=0)

test_gen = datagen.flow_from_dataframe(in_data[15000:],
                                       x_col="img_name", 
                                       y_col="comp",
                                       target_size=(224, 224),
                                       class_mode="binary",
                                       batch_size=32, 
                                       shuffle=True,
                                       seed=0)



print('Training network')

INIT_LR = PARAMS['learning_rate']
EPOCHS = PARAMS['n_epochs']
BS = PARAMS['batch_size']  

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(PARAMS['dense_units'], activation="relu")(headModel)
headModel = Dropout(PARAMS['dropout'])(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Log model summary
model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

# train the head of the network
H = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // BS,
        validation_data=validation_gen,
        validation_steps=validation_gen.n // BS,
        epochs=EPOCHS,
        callbacks=[LambdaCallback(on_epoch_end = lambda epoch, logs: log_data(logs)),
                   EarlyStopping(patience=PARAMS['early_stopping'], monitor='accuracy', restore_best_weights=True),
                   LearningRateScheduler(lr_scheduler)])


# ## Saving Model
# model.save(f"models/mouth_mobilenev2_model-{datetime.datetime.now()}.h5", save_format="h5")

print('Saving model')
# Log model weights
with tempfile.TemporaryDirectory(dir='.') as d:
    prefix = os.path.join(d, 'model_weights')
    model.save_weights(os.path.join(prefix, 'model'))
    for item in os.listdir(prefix):
        neptune.log_artifact(os.path.join(prefix, item),
                             os.path.join('model_weights', item))


# ### Testing Trained Model

# make predictions on the testing set
# predIdxs = model.predict(test_gen, batch_size=BS)
# predIdxs = np.argmax(predIdxs, axis=1)
# print(classification_report(test_gen.labels, predIdxs, target_names=['NON_COMP','COMP']))        

print('Evaluating model')

# Evaluate model
eval_metrics = model.evaluate(test_gen, verbose=0)
for j, metric in enumerate(eval_metrics):
    neptune.log_metric('eval_' + model.metrics_names[j], metric)


# ## Plot Training Curves

# plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure(figsize=(23,7))
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.ylim([0,1])
# plt.legend(loc="lower left")
# plt.savefig("figs/mouth_training_mobilenetv2.png")

print('Finishing Neptune')
neptune.stop()