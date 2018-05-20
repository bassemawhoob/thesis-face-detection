from IntegralImage import *
import os
import Utilities
from slidingwindow.helpers import pyramid
from slidingwindow.helpers import sliding_window
from PIL import Image


def mark_faces(image_path, cascade_path):
    # load the image and define the window width and height
    # image = load_images(path)[0]
    image = Image.open(image_path)
    (winW, winH) = (19, 19)
    # loop over the image pyramid
    cascade = Utilities.readFile(cascade_path)
    iterations = 1
    cur_scale = 1
    faces = []
    for resized in pyramid(image, scale=1.2):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=8, windowSize =(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.height != winH or window.width != winW:
                continue
            arr = np.array(window.convert('RGBA'))
            new_window = Image.fromarray(standardize_and_normalize(arr).astype('uint8'), 'RGBA').convert('L')
            iimage = IntegralImage(new_window, 1)
            pred = cascade.predict(iimage)
            if pred == 1 and skin_test(window):
                print("Face found here")
                window.save(os.path.join('result/images/', str(iterations) + '.jpg'))
                faces.append(tuple((x,y,iterations)))
            else:
                print("No faces found here")
        iterations += 1

def skin_test(image):
    image = image.convert("RGB")
    counter = 0
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r,g,b = image.getpixel((x,y))
            if g < r and b < r:
                counter = counter + 1
    if counter/(24**2):
        return True
    else:
        return False

def standardize_and_normalize(arr):
    std_arr = standardize_using_self(arr)
    nrm_arr = normalize(std_arr)
    return nrm_arr

def standardize_using_self(arr):
    arr = arr.astype('float')
    mean = np.mean(arr)
    std = np.std(arr)
    # No Changes in the Alpha Channel
    for i in range(3):
            arr[...,i] -= mean
            if std > 1: arr[...,i] /= std
    return arr

def standardize_using_parameters(arr):
    arr = arr.astype('float')
    mean_img = np.array(Image.open('trainingdata/parameters/set-average.jpg'))
    std_img = np.array(Image.open('trainingdata/parameters/set-std.jpg'))
    remove_mean = np.subtract(arr, mean_img)
    final = np.divide(remove_mean, std_img)
    return final

def normalize(arr):
    arr = arr.astype('float')
    # No Changes in the Alpha Channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr
