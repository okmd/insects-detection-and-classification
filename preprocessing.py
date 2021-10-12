import json
import logging
import os
import pickle
import re

import cv2
import numpy as np
import pandas as pd
import xmltodict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import gdown

import settings
import utils

# prepare all the necessary things before training.

# download dataset.
def download():
    drive_images_url = "https://drive.google.com/uc?id=1w7Lr4MQf03uREysDMh5Tr1T2xjfSxhRz"
    local_images_url = f"{settings.BASE_PATH}/images.tar"
    drive_annotation_url = "https://drive.google.com/uc?id=1TV1jaAp-XwhDupy_xfy4XPcmSFUb83c8"
    local_annotation_url = f"{settings.BASE_PATH}/annotations.tar"
    gdown.download(drive_images_url, local_images_url, quiet=False)
    utils.untar(local_images_url, settings.BASE_PATH)
    gdown.download(drive_annotation_url, local_annotation_url, quiet=False)
    utils.untar(local_annotation_url, settings.BASE_PATH)

# split the dataset into number of classes
def save_classes_pickle(save=False):
    labels = []
    file = os.path.join(settings.BASE_PATH, "classes.txt")
    content = utils.read(file)
    content = content.replace('\t', '').split('\n')
    for i in content:
        labels.append(("".join(re.split("[^a-zA-Z ]*", i)).strip()))
    # take first n classes only
    labels = labels[:settings.CLASSES]
    if save:
        utils.pickle_store(settings.CLASSES_PATH, labels)
        # utils.write(settings.CLASSES_PATH, pickle.dumps(labels), 'wb')
    return labels

# prepare cvs file of 1st n classes online
def prepare_row(file, classes):
    xml_content = utils.read(file)
    xml_dictionary = json.loads(json.dumps(xmltodict.parse(xml_content)))
    image_name = xml_dictionary['annotation']['filename']
    ans = []
    objs = xml_dictionary['annotation']['object']
    def get_min_max(obj):
        xmin = obj['bndbox']['xmin']
        ymin = obj['bndbox']['ymin']
        xmax = obj['bndbox']['xmax']
        ymax = obj['bndbox']['ymax']
        return xmin, ymin, xmax, ymax
        
    if isinstance(objs, list):
        for obj in objs:
            if int(obj['name']) >= settings.CLASSES:
                break
            classname = classes[int(obj['name'])].title()
            xmin, ymin,xmax, ymax = get_min_max(obj)
            ans.append((image_name, xmin, ymin, xmax,ymax, classname))
    
    elif int(objs['name']) < settings.CLASSES:
        classname =  classes[int(objs['name'])].title()
        xmin, ymin,xmax, ymax = get_min_max(objs)
        ans.append((image_name, xmin, ymin, xmax,ymax, classname))
    
    return ans

def save_csv(classes, save=False):
    for r, f, files in os.walk(settings.ANNOTATION_PATH):
        pass
    csv_files = []
    for file in tqdm(files):
        try:
            for row in prepare_row(f"{settings.ANNOTATION_PATH}/{file}", classes):
                csv_files.append(row)
        except Exception as e:
            print(f"{settings.ANNOTATION_PATH}/{file}")
            print(e)
    if save:
        cols = ['filename', 'xmin', 'ymin','xmax', 'ymax', 'class']
        pdf = pd.DataFrame(csv_files, columns=cols)
        pdf.to_csv(settings.COMPLETE_PATH, index=None)
        # utils.write(settings.COMPLETE_PATH, pickle.dumps(csv_files))
        

def prepare_data_pickle():
    data = []
    labels = []
    bboxes = []
    image_paths = []

    csv_file_content = utils.read(settings.COMPLETE_PATH)
    for row in tqdm(csv_file_content.split("\n")[1:]):
        try:
            (filename, startX, startY, endX, endY, label) = row.split(',')
            image_path = f"{settings.IMAGES_PATH}/{filename}"
            image = cv2.imread(image_path)
            h,w = image.shape[:2]
            sx = float(startX)/w
            sy= float(startY)/h
            ex = float(endX)/w
            ey = float(endY)/h
            image = load_img(image_path, target_size=(settings.IMG_WIDTH, settings.IMG_HEIGHT))
            image = img_to_array(image)
            data.append(image)
            labels.append(label)
            bboxes.append((sx,sy,ex,ey)) #(0.1523076923076923, 0.55, 0.8061538461538461, 0.7952380952380952)
            image_paths.append(image_path)
        except Exception as e:
            print(image_path)
            print(e)
    # convert to numpy 
    data = np.array(data, dtype=np.float32)/255.0
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype=np.float32)
    image_paths = np.array(image_paths)
    
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # split for trainging and testing
    split = train_test_split(data, labels, bboxes, image_paths, test_size=settings.TEST_SPLIT, random_state=42)
    tr_images, ts_images = split[:2]
    tr_labels, ts_labels = split[2:4]
    tr_bboxs, ts_bboxs = split[4:6]
    tr_img_paths, ts_img_paths = split[6:]
    # save to disk
    utils.pickle_store(settings.TRAIN_IMAGE_PICKLE_URL, tr_images)
    utils.pickle_store(settings.TEST_IMAGE_PICKLE_URL, ts_images)
    utils.pickle_store(settings.TRAIN_LABELS_PICKLE_URL, tr_labels)
    utils.pickle_store(settings.TEST_LABELS_PICKLE_URL, ts_labels)
    utils.pickle_store(settings.TRAIN_BBOX_PICKLE_URL, tr_bboxs)
    utils.pickle_store(settings.TEST_BBOX_PICKLE_URL, ts_bboxs)
    utils.pickle_store(settings.TRAIN_IMAGE_PATHS_PICKLE_URL, tr_img_paths)
    utils.pickle_store(settings.TEST_IMAGE_PATHS_PICKLE_URL, ts_img_paths)
    utils.pickle_store(settings.LB_PATH, lb)

# prepare and save into the pickel file.

# Create necessary folders
utils.create_directories()
logging.info("Downloading Data.")
download()
logging.info("Saving the class names in pickel.")
classes = save_classes_pickle(save=True)
logging.info("Creating Info Csv file for all data of n classes and their images.")
save_csv(classes, save=True)
logging.info("Creating pickle file of images, labels, bboxes and image_paths.")
prepare_data_pickle()
