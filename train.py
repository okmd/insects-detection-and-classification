# This script for training the model provided the thing are in place.
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import model
import settings
import utils

model = model.classification_regression_model

# data = pickle.loads(utils.read(settings.IMAGE_PICKLE_URL, 'rb'))
# labels = pickle.loads(utils.read(settings.LABELS_PICKLE_URL, 'rb'))
# bboxes = pickle.loads(utils.read(settings.BBOX_PICKLE_URL, 'rb'))
# image_paths = pickle.loads(utils.read(settings.IMAGE_PATHS_PICKLE_URL, 'rb'))
# convert all the list into the numpy array
# data = np.array(data, dtype=np.float32)/255.0
# labels = np.array(labels)
# bboxes = np.array(bboxes, dtype=np.float32)
# image_paths = np.array(image_paths)
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
tr_images = utils.pickle_load(settings.TRAIN_IMAGE_PICKLE_URL)
ts_images =utils.pickle_load(settings.TEST_IMAGE_PICKLE_URL)
tr_labels = utils.pickle_load(settings.TRAIN_LABELS_PICKLE_URL)
ts_labels =utils.pickle_load(settings.TEST_LABELS_PICKLE_URL)
tr_bboxs =utils.pickle_load(settings.TRAIN_BBOX_PICKLE_URL)
ts_bboxs = utils.pickle_load(settings.TEST_BBOX_PICKLE_URL)
tr_img_paths = utils.pickle_load(settings.TRAIN_IMAGE_PATHS_PICKLE_URL)
ts_img_paths = utils.pickle_load(settings.TEST_IMAGE_PATHS_PICKLE_URL)
lb = utils.pickle_load(settings.LB_PATH)
# split for trainging and testing
# split = train_test_split(data, labels, bboxes,
#                          image_paths, test_size=0.2, random_state=42)
# tr_images, ts_images = split[:2]
# tr_labels, ts_labels = split[2:4]
# tr_bboxs, ts_bboxs = split[4:6]
# tr_img_paths, ts_img_paths = split[6:]


losses = {
    # for multi-class classification
    "class_label": "categorical_crossentropy",
    # for regression or 4 points of bbox
    "bounding_box": "mean_squared_error",
}
# define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
loss_weights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}
train_targets = {
    "class_label": tr_labels,
    "bounding_box": tr_bboxs
}
# construct a second dictionary, this one for our target testing
# outputs
test_targets = {
    "class_label": ts_labels,
    "bounding_box": ts_bboxs
}
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# train_generator = train_datagen.flow(
#     tr_images,
#     batch_size = settings.BATCH_SIZE,
#     )
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=settings.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=[
              "accuracy"], loss_weights=loss_weights)
print(model.summary())

cp_callback = ModelCheckpoint(filepath=settings.CHEKPOINT_PATH, save_weights_only=True, verbose=1)
logging.info("Training the model.")
H = model.fit(
    tr_images, train_targets,
    validation_data=(ts_images, test_targets),
    batch_size=settings.BATCH_SIZE,
    epochs=settings.NUM_EPOCHS,
    callbacks=[cp_callback])

# save the model
logging.info("Saving the model.")
model.save(settings.MODEL_PATH, save_format="h5")
# save history
utils.pickle_store(settings.CHEKPOINT_PATH_HISTORY, H.history)
# save the label binarizer
# logging.info("Saving the label binarizer.")
# utils.write(settings.LB_PATH, pickle.dumps(lb), 'wb')



# save the image of training value.
utils.accuracy_plots(H.history)
utils.loss_plots(H.history)