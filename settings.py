import os
# Hyper-parameter for training.
INIT_LR = 0.0003
NUM_EPOCHS = 20
BATCH_SIZE = 32
CLASSES = 10
TEST_SPLIT = 0.2
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL = 224, 224, 3

# Input directory
BASE_DIR = os.getcwd()
BASE_PATH = os.path.join(BASE_DIR, "data/ip102/detection/")
IMAGES_PATH = os.path.join(BASE_PATH, "JPEGImages")
ANNOTATION_PATH = os.path.join(BASE_PATH, "Annotations")
CLASSES_PATH = os.path.join(BASE_PATH, "classes.pickle")
COMPLETE_PATH = os.path.join(BASE_PATH, "complete.csv")
NORMALIZED_PATH = os.path.join(BASE_PATH, "normalized.csv")


# Output directory
BASE_OUTPUT = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_OUTPUT, 'detection.h5')
PLOTS_PATH = os.path.join(BASE_OUTPUT, "plots")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")
CHEKPOINT_PATH = os.path.join(BASE_OUTPUT, 'checkpoints')


## PICKLES
PICKLE_URL = os.path.join(BASE_OUTPUT, "pickles")
TRAIN_IMAGE_PICKLE_URL = os.path.join(PICKLE_URL, "train_images.pickle")
TEST_IMAGE_PICKLE_URL = os.path.join(PICKLE_URL, "test_images.pickle")
TRAIN_LABELS_PICKLE_URL = os.path.join(PICKLE_URL, "train_labels.pickle")
TEST_LABELS_PICKLE_URL = os.path.join(PICKLE_URL, "test_labels.pickle")
TRAIN_BBOX_PICKLE_URL = os.path.join(PICKLE_URL, "train_bbox.pickle")
TEST_BBOX_PICKLE_URL = os.path.join(PICKLE_URL, "test_bbox.pickle")
TRAIN_IMAGE_PATHS_PICKLE_URL = os.path.join(PICKLE_URL, "train_image_paths.pickle")
TEST_IMAGE_PATHS_PICKLE_URL = os.path.join(PICKLE_URL, "test_image_paths.pickle")
LB_PATH = os.path.join(PICKLE_URL, "lb.pickle")
CHEKPOINT_PATH_HISTORY = os.path.join(PICKLE_URL, 'history.pickle')