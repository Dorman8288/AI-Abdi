from Utils import Counter, flipCoin
from time import sleep
import math
from numpy import random
from Enviroments import Enviroment
import numpy as np
from featureExtractors import FeatureExtractor
import pickle
class Agent:
    def __init__(self) -> None:
        pass

    def ChooseAction(self):
        pass

    def Train(self, EpisodeCount):
        pass

    def Test(self, EpisodeCount):
        pass


class QlearningAgent(Agent):
    def __init__(self, env: Enviroment):
        self.qValues = Counter()
        self.env = env

    def Save(self, Name):
        address = f"{Name}.pkl"
        with open(address, 'wb') as file:
            pickle.dump(self.qValues, file)
        print(f'Object successfully saved to "{address}"')

    def Load(self, Name):
        address = f"{Name}.pkl"
        with open(address, "rb") as file:
            self.qValues = pickle.load(file)
        print(f"{address} Loaded")

    def getQValue(self, state, action):
        return self.qValues[state, action]

    def computeValueFromQValues(self, state):
        actions = self.env.getPossibleActions()
        if len(actions) == 0:
            return 0.0
        return max([self.getQValue(state, action) for action in actions])

    def computeActionFromQValues(self, state):
        actions = self.env.getPossibleActions()
        candidates = []
        if len(actions) == 0:
          return None
        bestAction = None
        bestValue = -math.inf
        for action in actions:
            value = self.getQValue(state, action)
            if bestValue < value:
                bestValue = value
                bestAction = action
        return bestAction

    def ChooseAction(self, state):
        legalActions = self.env.getPossibleActions()
        action = None
        if flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward: float, ended):
      actions = self.env.getPossibleActions()
      sampleValue = reward
      if not ended:
        sampleValue += max([self.discount * self.getQValue(nextState, action) for action in actions])
      self.qValues[state, action] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sampleValue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def Train(self, EpisodeCount, discount, alpha, epsilon, epsilonRate, render, frameRate):
        self.discount = discount

        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilonRate = epsilonRate
        epsilonUpdateCount = math.log(0.1 / epsilon, self.epsilonRate)
        self.epsionUpdatePeriod = EpisodeCount // epsilonUpdateCount
        print(self.epsionUpdatePeriod)
        for _ in range(EpisodeCount):
            ended = False
            while True:
                prevState = self.env.GetCurrentState()
                action = self.ChooseAction(prevState)
                reward, terminated, truncated, info = self.env.doAction(action)
                currState = self.env.GetCurrentState()
                ended = truncated or terminated
                self.update(prevState, action, currState, reward, ended)
                if render:
                    self.env.render()
                    sleep(1/frameRate)
                if ended:
                    self.env.reset()
                    break
            if _ != 0 and _ % self.epsionUpdatePeriod == 0:
                print(self.epsilon)
                self.epsilon *= self.epsilonRate 

    def Test(self, EpisodeCount, frameRate, testEnv: Enviroment):
        for _ in range(EpisodeCount):
            while True:
                prevState = testEnv.GetCurrentState()
                action = self.getPolicy(prevState)
                reward, terminated, truncated, info = testEnv.doAction(action)
                ended = truncated or terminated
                testEnv.render()
                sleep(1/frameRate)
                if ended:
                    testEnv.reset()
                    break


class ApproximateQAgent(Agent):
    def __init__(self, extractor: FeatureExtractor, enviroment: Enviroment):
        self.featExtractor = extractor
        self.env = enviroment
        self.weights = Counter()
        feats = self.featExtractor.getFeatures(self.env.GetCurrentState(), 0)
        for feat in feats:
            self.weights[feat] = 1.0

    def Save(self, Name):
        address = f"{Name}.pkl"
        with open(address, 'wb') as file:
            pickle.dump(self.weights, file)
        print(f'Object successfully saved to "{address}"')

    def Load(self, Name):
        address = f"{Name}.pkl"
        with open(address, "rb") as file:
            self.weights = pickle.load(file)
        print(f"{address} Loaded")

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        feats = self.featExtractor.getFeatures(state, action)
        #print( feats * self.weights)
        return self.weights * feats

    def update(self, state, action, nextState, reward: float, ended):
        feats = self.featExtractor.getFeatures(state, action)
        updatedWeights = {}
        for feat in feats:
          updatedWeights[feat] = self.weights[feat]
          nextStateValue = self.getValue(nextState) if not ended else 0
          diffrence = (reward + self.discount * nextStateValue) - self.getQValue(state, action)
          updatedWeights[feat] += self.alpha * diffrence * feats[feat]
        for feat in updatedWeights:
            self.weights[feat] = updatedWeights[feat]

    def computeValueFromQValues(self, state):
        actions = self.env.getPossibleActions()
        if len(actions) == 0:
            return 0.0
        return max([self.getQValue(state, action) for action in actions])

    def computeActionFromQValues(self, state):
        actions = self.env.getPossibleActions()
        candidates = []
        if len(actions) == 0:
          return None
        bestAction = None
        bestValue = -math.inf
        for action in actions:
            value = self.getQValue(state, action)
            if bestValue < value:
                bestValue = value
                bestAction = action
        return bestAction

    def ChooseAction(self, state):
        legalActions = self.env.getPossibleActions()
        action = None
        if flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.computeActionFromQValues(state)
        return action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def Train(self, EpisodeCount, discount, alpha, epsilon, epsilonRate, render, frameRate):
        self.discount = discount

        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilonRate = epsilonRate
        epsilonUpdateCount = math.log(0.1 / epsilon, self.epsilonRate)
        self.epsionUpdatePeriod = EpisodeCount // epsilonUpdateCount
        print(self.epsionUpdatePeriod)
        for _ in range(EpisodeCount):
            ended = False
            while True:
                prevState = self.env.GetCurrentState()
                action = self.ChooseAction(prevState)
                reward, terminated, truncated, info = self.env.doAction(action)
                currState = self.env.GetCurrentState()
                ended = truncated or terminated
                self.update(prevState, action, currState, reward, ended)
                if render:
                    self.env.render()
                    sleep(1/frameRate)
                if ended:
                    self.env.reset()
                    break
            if _ != 0 and _ % self.epsionUpdatePeriod == 0:
                print(self.epsilon)
                self.epsilon *= self.epsilonRate 

    def Test(self, EpisodeCount, frameRate, testEnv: Enviroment):
        for _ in range(EpisodeCount):
            totalReward = 0
            while True:
                prevState = testEnv.GetCurrentState()
                action = self.getPolicy(prevState)
                reward, terminated, truncated, info = testEnv.doAction(action)
                totalReward += reward
                ended = truncated or terminated
                testEnv.render()
                sleep(1/frameRate)
                if ended:
                    testEnv.reset()
                    break
            print(totalReward)