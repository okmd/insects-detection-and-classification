# model
from tensorflow.keras.applications import vgg16, resnet
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D,MaxPooling2D
from tensorflow.keras.models import Model

import settings

# use = "vgg"
# if use=="vgg":
#     model = vgg16.VGG16(weights= "imagenet", include_top=False, input_tensor=Input(shape=(settings.IMG_WIDTH, settings.IMG_HEIGHT, settings.IMG_CHANNEL)))
# elif use=="resnet":
#     model = resnet.ResNet50(weights= "imagenet", include_top=False, input_tensor=Input(shape=(settings.IMG_WIDTH, settings.IMG_HEIGHT, settings.IMG_CHANNEL)))


def transfer_lr_model():
    model = vgg16.VGG16(weights= "imagenet", include_top=False, input_tensor=Input(shape=(settings.IMG_WIDTH, settings.IMG_HEIGHT, settings.IMG_CHANNEL)))
    model.trainable = False
    flatten = model.output
    flatten=Flatten()(flatten)

    bboxh = Dense(128, activation="relu")(flatten)
    bboxh = Dense(64, activation="relu")(bboxh)
    bboxh = Dense(32, activation="relu")(bboxh)
    bboxh = Dense(4, activation="sigmoid", name="bounding_box")(bboxh)

    softh= Dense(1024, activation="relu")(flatten)
    softh = Dropout(0.5)(softh)
    softh= Dense(512, activation="relu")(softh)
    softh = Dropout(0.5)(softh)
    softh = Dense(256, activation="relu")(softh)
    softh = Dropout(0.5)(softh)
    softh = Dense(settings.CLASSES, activation="softmax", name="class_label")(softh)
    model= Model(inputs=model.input, outputs=(bboxh, softh))
    print(model.summary())
    return model


def custom_model():
    inputs = Input(shape = (settings.IMG_WIDTH, settings.IMG_HEIGHT, settings.IMG_CHANNEL))
    conv1 = Conv2D(32, kernel_size = (5,5), strides = (1,1), activation = 'relu')(inputs)
    conv2 = Conv2D(32, kernel_size =(5, 5), strides =(1, 1),activation ='relu')(conv1)
    max1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv2)
    conv3 = Conv2D(32, kernel_size =(5, 5), strides =(1, 1),activation ='relu')(max1)
    max2 = MaxPooling2D(pool_size =(2, 2), strides =(2, 2))(conv3)
    conv4 = Conv2D(32, kernel_size =(5, 5), strides =(1, 1),activation ='relu')(max2)
    conv5 = Conv2D(32, kernel_size =(5, 5), strides =(1, 1),activation ='relu')(conv4)
    max3 = MaxPooling2D(pool_size=(2,2))(conv5)
    flat = Flatten()(max3)
    softh = Dense(settings.CLASSES, activation ='softmax', name="class_label")(flat)

    bboxh = Dense(128, activation="relu")(flat)
    bboxh = Dense(64, activation="relu")(bboxh)
    bboxh = Dense(32, activation="relu")(bboxh)
    bboxh = Dense(4, activation="sigmoid", name="bounding_box")(bboxh)
    model = Model(inputs = inputs, outputs=(bboxh, softh))
    print(model.summary())
    return  model

classification_regression_model = custom_model()