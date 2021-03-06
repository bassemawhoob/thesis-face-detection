import numpy as np
from AdaboostClassifier import AdaboostClassifier

class CascadeClassifier(object):

    def __init__(self):
        self.adaboostClassifiers = []
        self.falsePositiveRate = []
        self.detectionRate = []

    def addAdaboostClassifier(self, adaboostClassifier):
        self.adaboostClassifiers.append(adaboostClassifier)
        self.falsePositiveRate.append(1.0)
        self.detectionRate.append(1.0)

    def currentFPR(self):
        return self.falsePositiveRate[-1]

    def currentDR(self):
        return self.detectionRate[-1]

    def overallFPR(self):
        res = 1.0
        for fpr in self.falsePositiveRate:
            res *= fpr
        return res

    def prevFPR(self):
        if len(self.falsePositiveRate) < 2:
            return 1.0
        else:
            return self.falsePositiveRate[-2]

    def prevDR(self):
        if len(self.detectionRate) < 2:
            return 1.0
        else:
            return self.detectionRate[-2]

    def predict(self, image):
        for ada in self.adaboostClassifiers:
            if ada.predict(image) == 0:
                return 0
        return 1

    def evaluate(self, data):
        nPositive = 0
        nNegative = 0
        falseNegative = 0
        falsePositive = 0
        truePositive = 0
        trueNegative = 0
        for image in data:
            result = self.predict(image)
            if image.label == 1:
                nPositive += 1
                if result == 1:
                    truePositive += 1
                else:
                    falseNegative += 1
            else:
                nNegative += 1
                if result == 0:
                    trueNegative += 1
                else:
                    falsePositive += 1

        fpr = float(falsePositive / nNegative)
        dr = 1 - float(falseNegative / nPositive)

        self.falsePositiveRate[-1] = fpr
        self.detectionRate[-1] = dr

        recall = float(truePositive/(truePositive+falseNegative))
        accuracy = float((truePositive+trueNegative)/(truePositive+trueNegative+falseNegative+falsePositive))

        print("Recall:" + str(recall))
        print("Accuracy:" + str(accuracy))

        return fpr, dr

    def updateDataset(self, data, cascade):
        pos = []
        neg = []
        for image in data:
            if image.label == 0:
                if cascade.predict(image) == 1:
                    neg.append(image)
            if image.label == 1:
                if cascade.predict(image) == 1:
                    pos.append(image)
        return pos, neg

    def normalize_weights(self, posData, negData):
        pos_weight = 1. / (2 * len(posData))
        neg_weight = 1. / (2 * len(negData))
        for p in posData:
            p.set_weight(pos_weight)
        for n in negData:
            n.set_weight(neg_weight)

    def train(self, F, D, Ftarget, posData, negData, features):
        cascade = self
        self.normalize_weights(posData, negData)
        while self.overallFPR() > Ftarget:

            images = np.hstack((posData, negData))
            np.random.shuffle(images)

            ada = AdaboostClassifier()
            cascade.addAdaboostClassifier(ada)

            while cascade.currentFPR() > F * cascade.prevFPR():
                weakClassifier = ada.trainAdaboostClassifier(images, features)
                cascade.evaluate(images)
                while cascade.currentDR() < D * cascade.prevDR():
                    ada.decreaseThreshold()
                    cascade.evaluate(images)

            posData, negData = self.updateDataset(images, cascade)
        return cascade