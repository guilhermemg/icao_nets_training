import sys
import cv2
import random
import datetime
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if '../../../notebooks/' not in sys.path:
    sys.path.append('../../../notebooks/')

import utils.constants as cts
import utils.draw_utils as dr

from models.oface_mouth_model import OpenfaceMouth

from data_loaders.fvc_pyb_loader import FvcPybossaDL
from data_loaders.vgg_loader import VggFace2DL
from data_loaders.caltech_loader import CaltechDL
from data_loaders.cvl_loader import CvlDL
from data_loaders.colorferet_loader import ColorFeretDL
from data_loaders.fei_loader import FeiDB_DL
from data_loaders.gtech_loader import GeorgiaTechDL
from data_loaders.uni_essex_loader import UniEssexDL
from data_loaders.icpr04_loader import ICPR04_DL
from data_loaders.imfdb_loader import IMFDB_DL
from data_loaders.ijbc_loader import IJBC_DL
from data_loaders.lfw_loader import LFWDL
from data_loaders.celeba_loader import CelebA_DL
from data_loaders.casia_webface_loader import CasiaWebface_DL

from gt_loaders.gen_gt import Eval
from gt_loaders.fvc_gt import FVC_GTLoader
from gt_loaders.pybossa_gt import PybossaGTLoader

from tagger.tagger import Tagger

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    raise Exception("Invalid device or cannot modify virtual devices once initialized.")


m = OpenfaceMouth()

req = cts.ICAO_REQ.MOUTH

dl_list = [FvcPybossaDL(aligned=False), FvcPybossaDL(aligned=True),
           CaltechDL(aligned=False), 
           VggFace2DL(aligned=False), VggFace2DL(aligned=True),
           CvlDL(aligned=False), 
           ColorFeretDL(aligned=False), 
           FeiDB_DL(aligned=False), FeiDB_DL(aligned=True),
           CvlDL(aligned=False), 
           GeorgiaTechDL(aligned=False), GeorgiaTechDL(aligned=True), 
           UniEssexDL(aligned=False), 
           ICPR04_DL(aligned=False), 
           IMFDB_DL(aligned=True),
           IJBC_DL(aligned=False),
           LFWDL(aligned=False), LFWDL(aligned=True), 
           CelebA_DL(aligned=True),
           CasiaWebface_DL(aligned=False)
          ]

in_data = pd.DataFrame(columns=['origin','img_name','comp'])

for dl in dl_list:
    if dl.is_aligned():
        t = Tagger(dl, m, req)
        t.load_labels_df()
        tmp_df = t.labels_df
        tmp_df['origin'] = dl.get_name().value
        tmp_df['aligned'] = dl.is_aligned()
        in_data = in_data.append(tmp_df)

in_data['comp'] = in_data['comp'].astype('str')
in_data.shape    


# # Network Training

# ## Data Selection and Preprocessing
data = []
labels = []

for img_path,label in zip(in_data.img_name,in_data.comp):
	image = load_img(img_path, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

print(data.shape, labels.shape)


# ## Training InceptionV3

# In[15]:


datagen = ImageDataGenerator(preprocessing_function=prep_input_inceptionv3, 
                             validation_split=0.15)

train_gen = datagen.flow_from_dataframe(in_data[:15000], 
                                        x_col="img_name", 
                                        y_col="comp",
                                        target_size=(299, 299),
                                        class_mode="binary",
                                        batch_size=32, 
                                        shuffle=True,
                                        subset='training',
                                        seed=0)

validation_gen = datagen.flow_from_dataframe(in_data[:15000],
                                            x_col="img_name", 
                                            y_col="comp",
                                            target_size=(299, 299),
                                            class_mode="binary",
                                            batch_size=32, 
                                            shuffle=True,
                                            subset='validation',
                                            seed=0)

test_gen = datagen.flow_from_dataframe(in_data[15000:],
                                       x_col="img_name", 
                                       y_col="comp",
                                       target_size=(299, 299),
                                       class_mode="binary",
                                       batch_size=32, 
                                       shuffle=True,
                                       seed=0)


# In[16]:


INIT_LR = 1e-4
EPOCHS = 40
BS = 32  

# (trainX, testX, trainY, testY) = train_test_split(data, labels,
# 	test_size=0.20, stratify=labels, random_state=42)

# # construct the training image generator for data augmentation
# aug = ImageDataGenerator(
# 	rotation_range=20,
# 	zoom_range=0.15,
# 	width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	shear_range=0.15,
# 	horizontal_flip=True,
# 	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = InceptionV3(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(299, 299, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(8, 8))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	train_gen,
	steps_per_epoch=train_gen.n // BS,
	validation_data=validation_gen,
	validation_steps=validation_gen.n // BS,
	epochs=EPOCHS)


# ### Testing Trained Model

# In[17]:


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(test_gen, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(test_gen.labels, predIdxs, target_names=['NON_COMP','COMP']))


# ## Saving Model

# In[18]:


# serialize the model to disk
print("[INFO] saving model...")
model.save(f"models/mouth_inceptionv3_model-{datetime.datetime.now()}.h5", save_format="h5")


# ## Plot Training Curves

# In[19]:


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure(figsize=(23,7))
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.ylim([0,1])
plt.legend(loc="lower left")
plt.savefig("figs/mouth_training_inceptionv3.png")

