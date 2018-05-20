import numpy as np
from enum import Enum

# class FeatureType(Enum):
#     TWO_VERTICAL = (1, 2),
#     TWO_HORIZONTAL = (2, 1),
#     THREE_HORIZONTAL = (3, 1),
#     THREE_VERTICAL = (1, 3),
#     FOUR=(2, 2)

def enum(**enums):
    return type('Enum', (), enums)

FeatureType = enum(TWO_VERTICAL = (1,2), TWO_HORIZONTAL = (2,1), THREE_HORIZONTAL = (3,1), THREE_VERTICAL = (1,3), FOUR = (2,2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL,
                FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]

class HaarLikeFeature(object):

    def __init__(self, feature_type, position, width, height, threshold, polarity):
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity

    def generate_haar_features(self):
        features = []
        for f in FeatureTypes:
            for width in range(f[0], 20, f[0]):
                for height in range(f[1], 20, f[1]):
                    for x in range(20 - width):
                        for y in range(20 - height):
                            features.append(HaarLikeFeature(f, (x, y), width, height, 0, 1))
        return features

    def get_score(self, intImage):
        score = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = intImage.get_area_sum(self.top_left,
                                          (self.top_left[0] + self.width, self.top_left[1] + self.height // 2))
            second = intImage.get_area_sum((self.top_left[0], self.top_left[1] + self.height // 2), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = intImage.get_area_sum(self.top_left,
                                          (self.top_left[0] + self.width // 2, self.top_left[1] + self.height))
            second = intImage.get_area_sum((self.top_left[0] + self.width // 2, self.top_left[1]), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = intImage.get_area_sum(self.top_left,
                                          (self.top_left[0] + self.width // 3, self.top_left[1] + self.height))
            second = intImage.get_area_sum((self.top_left[0] + self.width // 3, self.top_left[1]),
                                           (self.top_left[0] + 2 * self.width // 3, self.top_left[1] + self.height))
            third = intImage.get_area_sum((self.top_left[0] + 2 * self.width // 3, self.top_left[1]), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = intImage.get_area_sum(self.top_left, (self.bottom_right[0], self.top_left[1] + self.height // 3))
            second = intImage.get_area_sum((self.top_left[0], self.top_left[1] + self.height // 3),
                                           (self.bottom_right[0], self.top_left[1] + 2 * self.height // 3))
            third = intImage.get_area_sum((self.top_left[0], self.top_left[1] + 2 * self.height // 3),
                                          self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            # top left area
            first = intImage.get_area_sum(self.top_left,
                                          (self.top_left[0] + self.width // 2, self.top_left[1] + self.height // 2))
            # top right area
            second = intImage.get_area_sum((self.top_left[0] + self.width // 2, self.top_left[1]),
                                           (self.bottom_right[0], self.top_left[1] + self.height // 2))
            # bottom left area
            third = intImage.get_area_sum((self.top_left[0], self.top_left[1] + self.height // 2),
                                          (self.top_left[0] + self.width // 2, self.bottom_right[1]))
            # bottom right area
            fourth = intImage.get_area_sum((self.top_left[0] + self.width // 2, self.top_left[1] + self.height // 2),
                                           self.bottom_right)
            score = first - second - third + fourth
        return score

    def get_vote(self, intImage):
        score = self.get_score(intImage)
        return 1 if score * self.polarity < self.polarity * self.threshold else 0

    def __str__(self):
        return ' FeatureType: {0} \n Position:{1} \n Width:{2} \n Height:{3} \n Threshold :{4} \n Polarity:{5}'.format(
            self.type, self.top_left, self.width, self.height, self.threshold, self.polarity)

    __repr__ = __str__

    def setThreshold(self, threshold):
        self.threshold = threshold

    def setPolarity(self, polarity):
        self.polarity = polarity



