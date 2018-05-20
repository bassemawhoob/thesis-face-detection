import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class AdaboostClassifier(object):

    def __init__(self):
        self.threshold = 0.0
        self.weakClassifiers = []

    def increaseThreshold(self, alpha):
        self.threshold += (0.5 * alpha)

    def decreaseThreshold(self, step=0.01):
        self.threshold -= step

    def addWeakClassifier(self, weakClassifier):
        self.weakClassifiers.append(weakClassifier)
        # Parameter 0 is the alpha (weight)
        self.increaseThreshold(weakClassifier[0])

    def predict(self, image):
        return 1 if sum([c[1].get_vote(image) * c[0] for c in self.weakClassifiers]) >= self.threshold else 0

    def trainAdaboostClassifier(self, images, features):
        # normalize the weights
        self.normalizeWeight(images)

        # select he best weak classifier with respect to the weighted error
        error, best = self.bestStump_multithread(features, images)

        # don't select a feature twice
        features.remove(best)
        print('<< error : {0} >>'.format(error))

        # update weights
        for image in images:
            if image.label == best.get_vote(image):
                image.set_weight(image.weight * np.sqrt(error / (1 - error)))
            else:
                image.set_weight(image.weight * np.sqrt((1 - error) / error))

        # final strong classifier
        alpha = np.log((1 - error) / error)

        weak_classifier = (alpha, best)

        self.addWeakClassifier(weak_classifier)

        return weak_classifier

    def normalizeWeight(self, images):
        norm_factor = 1. / sum(map(lambda image: image.weight, images))
        for image in images:
            image.set_weight(image.weight * norm_factor)

    def bestStump(self,features, images):
        error = float('inf')
        bestFeautre = None

        count = 1
        for f in features:
            count += 1

            e = self.decisionStump(f, images)
            if e < error:
                bestFeautre = f
                error = e

        if error >= 0.5:
            print('< Decision stump failed : best error ({0}) >= 0.5 >'.format(error))

        return error, bestFeautre

    def bestStump_multithread(self, features, images):
        # Since we're computing the minimum, we need an upper bound; thus the infinity
        error = float('inf')
        bestFeature = None

        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = {executor.submit(self.decisionStump, f, images): f for f in features}
            for future in concurrent.futures.as_completed(futures):
                # extract the Feature from the futures set
                f = futures[future]
                try:
                    e = future.result()
                    if e < error:
                        bestFeature = f
                        error = e
                except Exception as exc:
                    print(exc)

        if error >= 0.5:
            print('< Decision stump failed : best error ({0}) >= 0.5 >'.format(error))

        return error, bestFeature

    def decisionStump(self,feature, images):

        featureScore = []
        for image in images:
            featureScore.append((feature.get_score(image), image))

        featureScore = np.array(featureScore)
        # sorted featureScore according feature score

        featureScore = featureScore[featureScore[:, 0].argsort()]

        # total sum of positive weight
        tpw = 0.0
        # total sum of negative weight
        tnw = 0.0
        # sum of positive weight below current example
        pw = []
        # sum of negative weight below current example
        nw = []

        n = 0
        for f in featureScore:
            if f[1].label > 0:
                tpw += f[1].weight
            else:
                tnw += f[1].weight
            pw.append(tpw)
            nw.append(tnw)
            n += 1

        error = float("inf")
        polarity = 1
        threshold = featureScore[0][0]

        for i in range(n):
            pos = 0.0
            neg = 0.0
            if featureScore[i][1].label > 0:
                pos = (pw[i] - featureScore[i][1].weight) + (tnw - nw[i])
                neg = nw[i] + (tpw - pw[i] + featureScore[i][1].weight)
            else:
                pos = pw[i] + (tnw - nw[i] + featureScore[i][1].weight)
                neg = (nw[i] - featureScore[i][1].weight) + (tpw - pw[i])

            # e = min (S+  + (T- - S-), S- + (T+ - S+))
            # labelling all example below current negative and labeling  above positive
            # or the converse
            curError = 0.0
            curPolarity = 0
            if pos < neg:  # negative | threshold | postive
                curPolarity = -1
                curError = pos
            else:  # positive | threshold | negative
                curPolarity = 1
                curError = neg

            if error > curError:
                threshold = featureScore[i][0]
                error = curError
                polarity = curPolarity

        #  negative | threshold
        if tpw < error:
            error = tpw
            polarity = -1
            threshold = featureScore[0][0]
        # positive | threshold
        if tnw < error:
            error = tnw
            polarity = 1
            threshold = featureScore[0][0]

        feature.setThreshold(threshold)
        feature.setPolarity(polarity)

        return error
