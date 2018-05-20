# # %%% TRAINING PHASE IMPORTS %%%
import Utilities as utils
import random
import string
from HaarLikeFeature import HaarLikeFeature
from CascadeClassifier import CascadeClassifier
import math
import time
import numpy as np
# # %%% PREDICITION PHASE IMPORTS %%%
import Predictor

if __name__ == "__main__":

    # # %%% TRAINING PHASE %%%
    # # Load data
    # faces = utils.load_images('trainingdata/faces', 1)
    # non_faces = utils.load_images('trainingdata/nonfaces', 0)
    # # Generate Features
    # features = HaarLikeFeature.generate_haar_features()
    # # Build and train cascade
    # cascade = CascadeClassifier()
    # # False Positive Rate = 0.5 and Detection Rate = 0.99 from the analysis paper 10^-6 is the FPR target
    # cascade = cascade.train(0.5, 0.99, math.pow(10, -6), faces, non_faces, features)
    # # Save the output cascade from the training phase
    # # Generate a random 8 string name
    # name = 'result/' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) + '.txt'
    # utils.writeOut(cascade, name)

    # # %%% EVALUATE FINAL CASCADE %%%
    # faces = utils.load_images('trainingdata/faces', 1)
    # non_faces = utils.load_images('trainingdata/nonfaces', 0)
    # cascade = utils.readFile('result/trial100/cascade.txt')
    # images = np.hstack((faces, non_faces))
    # np.random.shuffle(images)
    # cascade.evaluate(images)

    #  %%% PREDICITION PHASE %%%
    Predictor.mark_faces('try/10.jpg', 'result/trial100/cascade.txt')







