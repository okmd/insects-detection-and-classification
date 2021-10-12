import os
import pickle
import tarfile
import matplotlib.pyplot as plt
import numpy as np

import settings


def read(file, mode='r'):
    with open(file, mode) as f:
        content = f.read()
    return content

def write(file, content, mode='w'):
    with open(file, mode) as f:
        f.write(content)

def pickle_load(file):
    return pickle.loads(read(file, 'rb'))

def pickle_store(file, content):
    write(file, pickle.dumps(content), 'wb')


def create_directories():
    os.makedirs(settings.BASE_OUTPUT, exist_ok=True)
    os.makedirs(settings.PLOTS_PATH, exist_ok=True)
    os.makedirs(settings.PICKLE_URL, exist_ok=True)

def untar(file, output):
    my_tar = tarfile.open(file)
    my_tar.extractall(output) # specify which folder to extract to
    my_tar.close()


def accuracy_plots(history):
    N = np.arange(0, settings.NUM_EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history["class_label_accuracy"], label="class_label_train_acc")
    plt.plot(N, history["bounding_box_accuracy"], label="bounding_box_train_acc")
    plt.plot(N, history["val_class_label_accuracy"], label="val_class_label_acc")
    plt.plot(N, history["val_bounding_box_accuracy"], label="val_bounding_box_acc")
    plt.title("Class Label Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    # save the accuracies plot
    plotPath = os.path.join(settings.PLOTS_PATH, f"accs_{settings.NUM_EPOCHS}.png")
    plt.savefig(plotPath)


def loss_plots(history):
    N = np.arange(0, settings.NUM_EPOCHS)
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
        ax[i].plot(N, history[l], label=l)
        ax[i].plot(N, history["val_" + l], label="val_" + l)
        ax[i].legend()
    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plotPath = os.path.join(settings.PLOTS_PATH, f"losses_{settings.NUM_EPOCHS}.png")
    plt.savefig(plotPath)
    plt.close()