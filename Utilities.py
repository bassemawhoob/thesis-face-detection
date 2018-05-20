import os
import IntegralImage
import sys
from PIL import Image

def load_images(path, label):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.jpg') or _file.endswith('.pgm') or _file.endswith('.png'):
            temp = Image.open(os.path.join(path, _file))
            images.append(IntegralImage.IntegralImage(temp, label))
    return images


def writeOut(cascade, file):
    os.makedirs(os.path.dirname(file))
    with open(file, 'wb') as f:
        import pickle
        pickle.dump(cascade, f)


def readFile(file):
    import pickle
    with open(file, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    return obj


def debug(message):
    sys.stdout.write(message)
    sys.stdout.flush()
