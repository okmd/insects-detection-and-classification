# model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model

import settings

vgg = VGG16(weights= "imagenet", include_top=False, input_tensor=Input(shape=(settings.IMG_WIDTH, settings.IMG_HEIGHT, settings.IMG_CHANNEL)))
vgg.trainable = False
flatten = vgg.output
flatten=Flatten()(flatten)

bboxh = Dense(128, activation="relu")(flatten)
bboxh = Dense(64, activation="relu")(bboxh)
bboxh = Dense(32, activation="relu")(bboxh)
bboxh = Dense(4, activation="sigmoid", name="bounding_box")(bboxh)

softh= Dense(512, activation="relu")(flatten)
softh = Dropout(0.5)(softh)
softh = Dense(512, activation="relu")(softh)
softh = Dropout(0.5)(softh)
softh = Dense(settings.CLASSES, activation="softmax", name="class_label")(softh)

classification_regression_model = Model(inputs=vgg.input, outputs=(bboxh, softh))
