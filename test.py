
import pickle
import re

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import settings
import utils

model = load_model(settings.MODEL_PATH)
# model.load_weights(checkpoint_path)
ts_images =utils.pickle_load(settings.TEST_IMAGE_PICKLE_URL)
ts_labels =utils.pickle_load(settings.TEST_LABELS_PICKLE_URL)
ts_bboxs = utils.pickle_load(settings.TEST_BBOX_PICKLE_URL)
ts_img_paths = utils.pickle_load(settings.TEST_IMAGE_PATHS_PICKLE_URL)
lb = utils.pickle_load(settings.LB_PATH)

test_targets = {
    "class_label": ts_labels,
    "bounding_box": ts_bboxs
}
loss, bbox_lox, label_loss, bbox_acc, label_acc = model.evaluate(ts_images, test_targets, verbose=1)
print("Untrained model, box accuracy: {:5.2f}%".format(100 * bbox_acc))
print("Untrained model, label accuracy: {:5.2f}%".format(100 * label_acc))


# 7/7 - 52s - loss: 1.9164 - bounding_box_loss: 0.0139 - class_label_loss: 1.9025 - bounding_box_accuracy: 0.7296 - class_label_accuracy: 0.4082
# Traceback (most recent call last):
#   File ".\test.py", line 27, in <module>
#     loss, acc = model.evaluate(ts_images, test_targets, verbose=2)
# ValueError: too many values to unpack (expected 2)





def predict(image_path):
    image = load_img(image_path, target_size=(
        settings.IMG_WIDTH, settings.IMG_HEIGHT))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    # predict the bounding box of the object along with the class
    # label
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]
    # determine the class label with the largest predicted
    # probability
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]
    return label, startX, startY, endX, endY


def draw_bbox_and_title(image_path):
    label, startX, startY, endX, endY = predict(image_path)
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # BGR image
    return image


def show(image):
    # show the output image
    # plt.grid(False)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imshow("A", image)
    cv2.waitKey(0)


image_path = f"{settings.IMAGES_PATH}/IP000000003.jpg"


# running
image = draw_bbox_and_title(image_path)
show(image)
