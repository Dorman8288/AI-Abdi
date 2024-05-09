
from Utils import Counter
import math

class FeatureExtractor:
    def getFeatures(self, state, action):
        pass

class CartPoleExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        features = Counter()
        features["angle"] = state[2]
        features["position"] = state[0]
        features["velocity"] = state[1]
        features["balancing"] = 1.0 if (state[3] > 0 and action == 1) or (state[3] < 0 and action == 0) else 0.0
        features["isDead"] = 1.0 if abs(features["angle"]) > 0.2 or abs(state[0]) > 2 else 0.0
        features["decelerating"] = (action == 1 and state[0] < 0) or (action == 0 and state[0] > 1)
        #print(features["2"])
        #features["accelerating"] = (action == 0 and state[1] < 0) or (action == 1 and state[1] > 1)
        return features
    

class LunarLanderExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        features = Counter()
        features["x"] = state[0]
        features["falling Speed"] = state[3]
        features["balancing"] = 1.0 if (state[0] > 0 and action == 3) or (state[0] < 0 and action == 1) else 0.0
        #features["needTime"] = 1.0 if (state[])
        return features
