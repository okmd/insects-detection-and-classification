
import pickle
import re
import argparse


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
lb = utils.pickle_load(settings.LB_PATH)

def evaluate():
    ts_images =utils.pickle_load(settings.TEST_IMAGE_PICKLE_URL)
    ts_labels =utils.pickle_load(settings.TEST_LABELS_PICKLE_URL)
    ts_bboxs = utils.pickle_load(settings.TEST_BBOX_PICKLE_URL)
    # ts_img_paths = utils.pickle_load(settings.TEST_IMAGE_PATHS_PICKLE_URL)
    test_targets = {
        "class_label": ts_labels,
        "bounding_box": ts_bboxs
    }
    loss, bbox_loss, label_loss, bbox_acc, label_acc = model.evaluate(ts_images, test_targets, verbose=1)
    print("Total Loss: {:5.2f}%".format(100 * loss))
    print("Label Loss: {:5.2f}%".format(100 * label_loss))
    print("Label Accuracy: {:5.2f}%".format(100 * label_acc))
    print("Box Loass: {:5.2f}%".format(100 * bbox_loss))
    print("Box Accuracy: {:5.2f}%".format(100 * bbox_acc))
    


def predict_box_label(image_path):
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
    print(label, i)
    return label, startX, startY, endX, endY, np.max(labelPreds)


def draw_bbox_and_title(image_path):
    label, startX, startY, endX, endY, prob = predict_box_label(image_path)
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
    prob = f"{prob*100:0.2f}"
    cv2.putText(image, f"{label} {prob}%", (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # BGR image
    return image, label, prob


def show(image):
    # show the output image
    # plt.grid(False)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"{settings.BASE_OUTPUT}/predictions.png", image)



# running
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process arguments to evaluate or predict.')
    parser.add_argument('-e', '--evaluate', action="store_true", help='Use to evaluate the model.')
    parser.add_argument('-p', '--predict',  help='Provide the image to predict and prediction will be saved in output as predictions.png.')

    args = parser.parse_args()



    image_path = f"{settings.IMAGES_PATH}/IP000000003.jpg"

    if args.evaluate:
        evaluate()
    if args.predict:
        image = draw_bbox_and_title(args.predict)
        show(image)
