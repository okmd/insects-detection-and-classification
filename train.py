# This script for training the model provided the thing are in place.
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
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
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=settings.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=[
              "accuracy"], loss_weights=loss_weights)
print(model.summary())

cp_callback = ModelCheckpoint(filepath=settings.CHEKPOINT_PATH,save_weights_only=True, verbose=1)
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

# save the label binarizer
# logging.info("Saving the label binarizer.")
# utils.write(settings.LB_PATH, pickle.dumps(lb), 'wb')



# save the image of training value.
N = np.arange(0, settings.NUM_EPOCHS)
def accuracy_plots():
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["class_label_accuracy"],
        label="class_label_train_acc")
    plt.plot(N, H.history["val_class_label_accuracy"],
        label="val_class_label_acc")
    plt.title("Class Label Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    # save the accuracies plot
    plotPath = os.path.join(settings.PLOTS_PATH, "accs.png")
    plt.savefig(plotPath)


def loss_plots():
    lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
    # loop over the loss names
    for (i, l) in enumerate(lossNames):
        # plot the loss for both the training and validation data
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(N, H.history[l], label=l)
        ax[i].plot(N, H.history["val_" + l], label="val_" + l)
        ax[i].legend()
    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plotPath = os.path.join(settings.PLOTS_PATH, "losses.png")
    plt.savefig(plotPath)
    plt.close()


accuracy_plots()
loss_plots()