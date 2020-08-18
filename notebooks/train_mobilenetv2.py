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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_input_mobilenetv2
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

# from gt_loaders.gen_gt import Eval
# from gt_loaders.fvc_gt import FVC_GTLoader
# from gt_loaders.pybossa_gt import PybossaGTLoader


## restrict memory growth -------------------

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    raise Exception("Invalid device or cannot modify virtual devices once initialized.")

## restrict memory growth -------------------    


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


m = OpenfaceMouth()
req = cts.ICAO_REQ.MOUTH
dl_names = [x for x in DLName]
print(f'DL names: {dl_names}')

print('Loading data')
netDataLoader = NetDataLoader(m, req, dl_names, True)
in_data = netDataLoader.load_data()
print('Data loaded')


# # Network Training

N_TRAIN_PROP = 0.7
N_TEST_PROP = 0.3
N_TRAIN = int(len(in_data)*N_TRAIN_PROP)
N_TEST = len(in_data) - N_TRAIN
SEED = 42

print(f'N_TRAIN: {N_TRAIN}')
print(f'N_TEST: {N_TEST}')
print(f'N: {len(in_data)}')

# ## Training MobileNetV2

INIT_LR = 1e-4
EPOCHS = 40
BS = 64
SHUFFLE = True
DROPOUT = 0.5
EARLY_STOPPING = 10
OPTIMIZER = 'Adam'
DENSE_UNITS = 128

print('Starting data generators')
datagen = ImageDataGenerator(preprocessing_function=prep_input_mobilenetv2, 
                             validation_split=0.25,
                             rescale=1.0/255.0)

train_gen = datagen.flow_from_dataframe(in_data[:N_TRAIN], 
                                        x_col="img_name", 
                                        y_col="comp",
                                        target_size=(224, 224),
                                        class_mode="binary",
                                        batch_size=BS, 
                                        shuffle=SHUFFLE,
                                        subset='training',
                                        seed=SEED)

validation_gen = datagen.flow_from_dataframe(in_data[:N_TRAIN],
                                            x_col="img_name", 
                                            y_col="comp",
                                            target_size=(224, 224),
                                            class_mode="binary",
                                            batch_size=BS, 
                                            shuffle=SHUFFLE,
                                            subset='validation',
                                            seed=SEED)

test_gen = datagen.flow_from_dataframe(in_data[N_TRAIN:],
                                       x_col="img_name", 
                                       y_col="comp",
                                       target_size=(224, 224),
                                       class_mode="binary",
                                       batch_size=BS, 
                                       shuffle=False,
                                       seed=SEED)


# Define parameters
PARAMS = {'batch_size': BS,
          'n_epochs': EPOCHS,
          'shuffle': SHUFFLE,
          'dense_units': DENSE_UNITS,
          'learning_rate': INIT_LR,
          'optimizer': OPTIMIZER,
          'dropout': DROPOUT,
          'early_stopping': EARLY_STOPPING,
          'n_train_prop': N_TRAIN_PROP,
          'n_test_prop': N_TEST_PROP,
          'n_train': train_gen.n,
          'n_validation': validation_gen.n,
          'n_test': test_gen.n,
          'seed': SEED}


print('Creating experiment')
neptune.create_experiment(name='train_mobilenetv2',
                          params=PARAMS,
                          properties={'dl_names': str([dl_name.value for dl_name in dl_names]),
                                      'dl_aligned': True,
                                      'icao_req': req.value,
                                      'tagger_model': m.get_model_name().value},
                          description='Testing validation split equals to 0.25',
                          tags=['mobilenetv2'],
                          upload_source_files=['train_mobilenetv2.py'])


print('Training network')

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(PARAMS['dense_units'], activation="relu")(headModel)
headModel = Dropout(PARAMS['dropout'])(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

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
eval_metrics = model.evaluate(test_gen, verbose=0)
for j, metric in enumerate(eval_metrics):
    neptune.log_metric('eval_' + model.metrics_names[j], metric)


print('Finishing Neptune')
neptune.stop()