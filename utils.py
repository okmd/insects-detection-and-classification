import os
import pickle
import tarfile

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
