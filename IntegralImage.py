import numpy as np

class IntegralImage:

    def __init__(self, imageObj, label):
        keep = imageObj.copy()
        self.original = np.array(keep)
        self.image = keep
        self.sum = 0
        self.label = label
        self.calculate_integral()
        self.weight = 0
        # temp.close()

    def calculate_integral(self):
        rowSum = np.zeros(self.original.shape)
        self.integral = np.zeros((self.original.shape[0] + 1, self.original.shape[1] + 1))
        for x in range(self.original.shape[1]):
            for y in range(self.original.shape[0]):
                rowSum[y, x] = rowSum[y - 1, x] + self.original[y, x]
                self.integral[y + 1, x + 1] = self.integral[y + 1, x - 1 + 1] + rowSum[y, x]

    def get_area_sum(self, topLeft, bottomRight):
        topLeft = (topLeft[1], topLeft[0])
        bottomRight = (bottomRight[1], bottomRight[0])
        if topLeft == bottomRight:
            return self.integral[topLeft]
        topRight = (bottomRight[0], topLeft[1])
        bottomLeft = (topLeft[0], bottomRight[1])

        return self.integral[bottomRight] - self.integral[topRight] - self.integral[bottomLeft] + self.integral[topLeft]

    def set_label(self, label):
        self.label = label

    def set_weight(self, weight):
        self.weight = weight